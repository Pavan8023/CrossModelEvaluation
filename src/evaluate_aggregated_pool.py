"""Train one pooled model and evaluate it on the reserved test emails."""

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold

from evaluate_cross_dataset import feature_matrix
from evaluate_cross_model import CLASSIFIERS, GENERATORS
from evaluate_intra_model import calculate_metrics, llm_scores, save_search, selection_f1
from feature_validation import FEATURES, read_csv, sha256
from train_knn import ROOT, SEED, SPLITS, THRESHOLD, load_development, write_csv
from train_lightgbm import check_unchanged

N_FOLDS = 5


def prepare_plan():
    _, _, _, _, watched = load_development(GENERATORS[0])
    rows = sorted(read_csv(SPLITS / "manifest.csv"), key=lambda row: row["record_id"])
    train = [row for row in rows if row["partition"] == "development"]
    test = [row for row in rows if row["partition"] == "test"]

    groups = np.array([row["group_id"] for row in train])
    if set(groups) & {row["group_id"] for row in test}:
        raise ValueError("Development and test groups overlap.")

    for name, partition in (("Development", train), ("Test", test)):
        llm_sources = {row["source"] for row in partition if row["label"] == "1"}
        if llm_sources != set(GENERATORS) or not any(row["label"] == "0" for row in partition):
            raise ValueError(f"{name} must contain humans and all three generators.")
        print(name, "by source:", dict(Counter(row["source"] for row in partition)))

    y = np.array([int(row["label"]) for row in train])
    strata = [f"{row['source']}|{row['category']}" for row in train]
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(train)), strata, groups))
    assigned = np.full(len(train), -1, dtype=int)

    for fold, (fit, valid) in enumerate(folds):
        if (
            set(groups[fit]) & set(groups[valid])
            or set(y[fit]) != {0, 1}
            or set(y[valid]) != {0, 1}
            or len(fit) < 31
            or np.any(assigned[valid] != -1)
        ):
            raise ValueError(f"Invalid development fold {fold}.")
        assigned[valid] = fold
        print(f"Fold {fold}: train={len(fit)}, validation={len(valid)}, group overlap=0")

    if np.any(assigned == -1):
        raise ValueError("Missing development validation assignments.")
    return train, test, y, folds, assigned, watched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=CLASSIFIERS, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if THRESHOLD != 0.5:
        raise ValueError("The pooled evaluation uses a cutoff of 0.5.")

    detector = import_module(f"{args.classifier}_detector")
    factory = getattr(detector, f"build_{args.classifier}")
    grid = getattr(detector, f"{args.classifier}_parameter_grid")()
    n_candidates = len(ParameterGrid(grid))
    if n_candidates != 20:
        raise ValueError("Expected the existing 20-candidate grid.")

    train, test, y, folds, assigned, watched = prepare_plan()
    dependencies = [
        Path(detector.__file__).resolve(),
        ROOT / "src" / "evaluate_cross_dataset.py",
        ROOT / "src" / "evaluate_cross_model.py",
        ROOT / "src" / "evaluate_intra_model.py",
        ROOT / "src" / "train_knn.py",
        ROOT / "src" / "train_lightgbm.py",
    ]
    if not args.check_only:
        dependencies.append(Path(__file__).resolve())
    watched.update({path: sha256(path) for path in dependencies})
    check_unchanged(watched)

    if args.check_only:
        print(f"Preflight passed. Planned fits: {n_candidates * N_FOLDS + 1}.")
        return

    data = {row["id"]: row for row in read_csv(ROOT / "data" / "corpus_features.csv")}
    search = GridSearchCV(
        factory(), grid, scoring=selection_f1, cv=folds,
        refit=True, error_score="raise", n_jobs=1, verbose=1,
    )
    search.fit(feature_matrix(train, data), y)
    check_unchanged(watched)

    # Score each test email once. Reuse human scores in the generator breakdowns.
    scores = llm_scores(search.best_estimator_, feature_matrix(test, data))
    actual = np.array([int(row["label"]) for row in test])
    source = np.array([row["source"] for row in test])
    predicted = (scores >= THRESHOLD).astype(int)
    cells = []

    for scope in ("pooled", *GENERATORS):
        mask = np.ones(len(test), dtype=bool) if scope == "pooled" else (
            (actual == 0) | (source == scope)
        )
        cells.append({
            "test_scope": scope,
            "human_test_records": int(np.sum(mask & (actual == 0))),
            "llm_test_records": int(np.sum(mask & (actual == 1))),
            **calculate_metrics(actual[mask], scores[mask]),
        })

    run_id = (
        f"{args.classifier}_aggregated_pool_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    )
    output = ROOT / "results" / "aggregated_pool" / args.classifier / run_id
    output.mkdir(parents=True, exist_ok=False)

    predictions = [
        {
            "experiment_id": run_id,
            "record_id": row["record_id"],
            "group_id": row["group_id"],
            "email_source": row["source"],
            "category": row["category"],
            "actual_label": int(label),
            "llm_score": float(score),
            "predicted_label": int(prediction),
        }
        for row, label, score, prediction in zip(test, actual, scores, predicted)
    ]
    assignments = [
        {
            "record_id": row["record_id"],
            "group_id": row["group_id"],
            "validation_fold": int(fold),
        }
        for row, fold in zip(train, assigned)
    ]

    for name, records in (
        ("aggregated_pool_metrics.csv", cells),
        ("aggregated_pool_predictions.csv", predictions),
        ("aggregated_pool_cv_assignments.csv", assignments),
    ):
        write_csv(output / name, list(records[0]), records)

    save_search(output / "aggregated_pool_search.csv", search.cv_results_)
    joblib.dump(search.best_estimator_, output / "aggregated_pool_model.joblib")
    check_unchanged(watched)

    packages = ["numpy", "scikit-learn", "joblib"]
    if args.classifier != "knn":
        packages.append(args.classifier)

    report = {
        "experiment_id": run_id,
        "status": "aggregated_pool_complete",
        "evaluation_method": "aggregated_pool",
        "protocol_version": "pooled_development_v1",
        "classifier": args.classifier,
        "features": list(FEATURES),
        "split_version": "splits_v2",
        "threshold": THRESHOLD,
        "seed": SEED,
        "training_records": len(train),
        "development_source_counts": dict(Counter(row["source"] for row in train)),
        "best_parameters": search.best_params_,
        "development_cv_f1": float(search.best_score_),
        "fit_count": n_candidates * N_FOLDS + 1,
        "calibration_records_used": 0,
        "final_test_records_used": len(test),
        "prediction_rows": len(predictions),
        "cell_metrics": cells,
        "versions": {
            "python": platform.python_version(),
            **{package: metadata.version(package) for package in packages},
        },
        "input_hashes": {str(path): value for path, value in watched.items()},
        "result_directory": str(output),
        "output_hashes": {path.name: sha256(path) for path in sorted(output.iterdir())},
    }

    ledger = ROOT / "ledger" / "aggregated_pool" / args.classifier / f"{run_id}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    for cell in cells:
        print(f"{cell['test_scope']}: F1={cell['f1']:.4f}, FP={cell['fp']}, FN={cell['fn']}")
    print("Experiment record:", ledger)


if __name__ == "__main__":
    main()