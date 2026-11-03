"""Leave one human source out; keep pooled LLM train and test sets fixed."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold

from evaluate_cross_model import CLASSIFIERS, GENERATORS
from evaluate_intra_model import (
    calculate_metrics, llm_scores, save_search, selection_f1,
)
from feature_validation import FEATURES, read_csv, sha256
from train_knn import ROOT, SEED, SPLITS, THRESHOLD, load_development, write_csv
from train_lightgbm import check_unchanged


def prepare_plan():
    # Reuse the project's manifest, input-hash and linked-pair checks.
    _, _, _, _, watched = load_development(GENERATORS[0])
    rows = sorted(read_csv(SPLITS / "manifest.csv"), key=lambda r: r["record_id"])
    human_sources = sorted({r["source"] for r in rows if r["origin"] == "human"})
    llm_sources = {r["source"] for r in rows if r["origin"] == "llm"}
    if len(human_sources) != 5 or llm_sources != set(GENERATORS):
        raise ValueError("Unexpected human sources or LLM generators.")

    development = [r for r in rows if r["partition"] == "development"]
    test = [r for r in rows if r["partition"] == "test"]
    if any(not any(r["source"] == s for r in test) for s in human_sources):
        raise ValueError("Every human source needs a held-out test subset.")
    test_groups = {r["group_id"] for r in test}
    expected_llm = {r["record_id"] for r in development if r["origin"] == "llm"}

    plans, assignments = {}, []
    for held_out in human_sources:
        excluded = {r["group_id"] for r in rows if r["source"] == held_out}
        train = [r for r in development if r["group_id"] not in excluded]
        if {r["record_id"] for r in train if r["origin"] == "llm"} != expected_llm:
            raise ValueError("Human-source exclusion changed the fixed LLM pool.")
        human_train = {r["source"] for r in train if r["origin"] == "human"}
        if human_train != set(human_sources) - {held_out}:
            raise ValueError("Unexpected human sources in training.")

        groups = np.array([r["group_id"] for r in train])
        if set(groups) & test_groups:
            raise ValueError("Training and test groups overlap.")

        y = np.array([int(r["label"]) for r in train])
        strata = [f"{r['source']}|{r['category']}" for r in train]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = list(splitter.split(np.zeros(len(train)), strata, groups))
        assigned = np.full(len(train), -1, dtype=int)

        for fold, (fit, validation) in enumerate(folds):
            if (
                set(groups[fit]) & set(groups[validation])
                or set(y[fit]) != {0, 1}
                or set(y[validation]) != {0, 1}
                or len(fit) < 31
                or np.any(assigned[validation] != -1)
            ):
                raise ValueError(f"Invalid development fold for {held_out}.")
            assigned[validation] = fold

        if np.any(assigned == -1):
            raise ValueError("Missing validation assignments.")

        assignments.extend({
            "excluded_human_source": held_out,
            "record_id": r["record_id"],
            "group_id": r["group_id"],
            "validation_fold": int(fold),
        } for r, fold in zip(train, assigned))

        plans[held_out] = (train, y, folds)
        print(
            f"Exclude {held_out}: development human={int((y == 0).sum())}, "
            f"LLM={int(y.sum())}"
        )

    print("Test counts by source:", dict(Counter(r["source"] for r in test)))
    return human_sources, plans, test, assignments, watched


def feature_matrix(rows, features_by_id):
    X = np.array([
        [float(features_by_id[r["record_id"]][f]) for f in FEATURES]
        for r in rows
    ], dtype=float)
    if X.shape != (len(rows), len(FEATURES)) or not np.isfinite(X).all():
        raise ValueError("Invalid feature matrix.")
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=CLASSIFIERS, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if THRESHOLD != 0.5:
        raise ValueError("This protocol uses a fixed cutoff of 0.5.")

    detector = import_module(f"{args.classifier}_detector")
    factory = getattr(detector, f"build_{args.classifier}")
    grid = getattr(detector, f"{args.classifier}_parameter_grid")()
    n_candidates = len(ParameterGrid(grid))
    if n_candidates != 20:
        raise ValueError("Expected the existing 20-candidate grid.")

    sources, plans, test, assignments, watched = prepare_plan()
    dependencies = [
        Path(detector.__file__).resolve(),
        ROOT / "src" / "train_knn.py",
        ROOT / "src" / "evaluate_cross_model.py",
        ROOT / "src" / "evaluate_intra_model.py",
        ROOT / "src" / "train_lightgbm.py",
    ]
    if not args.check_only:
        dependencies.append(Path(__file__).resolve())
    watched.update({path: sha256(path) for path in dependencies})
    check_unchanged(watched)
    if args.check_only:
        print("Preflight passed for all five holdouts.")
        return

    run_id = (
        f"{args.classifier}_cross_dataset_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    )
    output = ROOT / "results" / "cross_dataset" / args.classifier / run_id
    output.mkdir(parents=True, exist_ok=False)
    data = {r["id"]: r for r in read_csv(ROOT / "data" / "corpus_features.csv")}
    searches, settings = {}, {}

    for held_out, (train, y, folds) in plans.items():
        print(f"Fitting {args.classifier} with {held_out} held out", flush=True)
        search = GridSearchCV(
            factory(), grid, scoring=selection_f1, cv=folds,
            refit=True, error_score="raise", n_jobs=1,
        )
        search.fit(feature_matrix(train, data), y)
        searches[held_out] = search.best_estimator_
        settings[held_out] = {
            "training_records": len(train),
            "training_human_records": int((y == 0).sum()),
            "training_llm_records": int(y.sum()),
            "best_parameters": search.best_params_,
            "development_cv_f1": float(search.best_score_),
        }
        save_search(output / f"search_without_{held_out}.csv", search.cv_results_)

    # All five models are fitted before any test predictions are made.
    check_unchanged(watched)
    X_test = feature_matrix(test, data)
    y_test = np.array([int(r["label"]) for r in test])
    test_sources = np.array([r["source"] for r in test])
    cells, predictions = [], []

    for held_out, model in searches.items():
        scores = llm_scores(model, X_test)
        predicted = (scores >= THRESHOLD).astype(int)

        for target in sources:
            mask = (y_test == 1) | (test_sources == target)
            results = calculate_metrics(y_test[mask], scores[mask])
            cells.append({
                "excluded_human_source": held_out,
                "test_human_source": target,
                "human_source_seen_in_training": target != held_out,
                "test_human_records": int(np.sum(mask & (y_test == 0))),
                "test_llm_records": int(y_test.sum()),
                "human_false_positive_rate": results["fp"] / (results["fp"] + results["tn"]),
                **results,
            })

        predictions.extend({
            "experiment_id": run_id,
            "excluded_human_source": held_out,
            "record_id": row["record_id"],
            "group_id": row["group_id"],
            "email_source": row["source"],
            "category": row["category"],
            "actual_label": int(actual),
            "llm_score": float(score),
            "predicted_label": int(pred),
        } for row, actual, score, pred in zip(test, y_test, scores, predicted))

    comparisons = []
    for source in sources:
        rates = {
            cell["excluded_human_source"]: cell["human_false_positive_rate"]
            for cell in cells if cell["test_human_source"] == source
        }
        unseen = rates[source]
        seen = float(np.mean([rate for excluded, rate in rates.items() if excluded != source]))
        comparisons.append({
            "human_source": source,
            "unseen_fpr": unseen,
            "mean_seen_fpr": seen,
            "unseen_minus_seen_fpr_pp": 100 * (unseen - seen),
        })

    for filename, records in (
        ("cross_dataset_metrics.csv", cells),
        ("cross_dataset_source_comparison.csv", comparisons),
        ("cross_dataset_predictions.csv", predictions),
        ("cross_dataset_cv_assignments.csv", assignments),
    ):
        write_csv(output / filename, list(records[0]), records)

    check_unchanged(watched)
    report = {
        "experiment_id": run_id,
        "status": "cross_dataset_complete",
        "evaluation_method": "cross_dataset",
        "protocol_version": "leave_one_human_source_out_v1",
        "classifier": args.classifier,
        "features": list(FEATURES),
        "split_version": "splits_v2",
        "threshold": THRESHOLD,
        "seed": SEED,
        "fit_count": len(sources) * (5 * n_candidates + 1),
        "calibration_records_used": 0,
        "final_test_records_used": len(test),
        "prediction_rows": len(predictions),
        "training_settings": settings,
        "source_comparison": comparisons,
        "cell_metrics": cells,
        "versions": {
            package: metadata.version(package)
            for package in ("numpy", "scikit-learn", "lightgbm", "catboost")
        },
        "input_hashes": {str(path): value for path, value in watched.items()},
        "result_directory": str(output),
        "output_hashes": {p.name: sha256(p) for p in sorted(output.iterdir())},
    }

    ledger = ROOT / "ledger" / "cross_dataset" / args.classifier / f"{run_id}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    for row in comparisons:
        print(
            f"{row['human_source']}: human flag rate "
            f"seen={row['mean_seen_fpr']:.2%}, unseen={row['unseen_fpr']:.2%}"
        )
    print("Experiment record:", ledger)


if __name__ == "__main__":
    main()