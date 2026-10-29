"""Development-only KNN selection using the shared version-2 partitions."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)

from feature_validation import FEATURES, read_csv, sha256
from knn_detector import (
    build_knn,
    knn_parameter_grid,
    predict_llm_scores,
)


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "experiments" / "splits_v2"
SEED = 42
THRESHOLD = 0.5


def write_csv(path, fields, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def load_development(generator):
    """Check provenance, then extract eligible development features."""

    summary_path = SPLITS / "summary.json"
    manifest_path = SPLITS / "manifest.csv"
    pairs_path = SPLITS / "similarity_pairs.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if (
        summary["status"] != "candidate_partitions_checks_passed"
        or summary["similarity_threshold"] != 0.88
    ):
        raise ValueError("Expected the checked version-2 split.")

    watched = {}

    for filename in (
        "corpus_features.csv",
        "human_corpus_sampled.csv",
        "llm_corpus_sampled.csv",
    ):
        path = ROOT / "data" / filename
        actual_hash = sha256(path)

        if actual_hash != summary["input_hashes"][filename]:
            raise ValueError(f"Input changed: {filename}")

        watched[path] = actual_hash

    for path in (
        summary_path,
        manifest_path,
        pairs_path,
        ROOT / "src" / "knn_detector.py",
        ROOT / "src" / "feature_validation.py",
    ):
        watched[path] = sha256(path)

    data = read_csv(ROOT / "data" / "corpus_features.csv")
    manifest = read_csv(manifest_path)

    data_by_id = {row["id"]: row for row in data}
    manifest_by_id = {row["record_id"]: row for row in manifest}

    if (
        len(data_by_id) != len(data)
        or len(manifest_by_id) != len(manifest)
        or set(data_by_id) != set(manifest_by_id)
    ):
        raise ValueError("Record IDs or manifest alignment are invalid.")

    if dict(Counter(row["partition"] for row in manifest)) != (
        summary["partition_counts"]
    ):
        raise ValueError("Partition counts differ from the split summary.")

    group_partitions = defaultdict(set)

    for row in manifest:
        original = data_by_id[row["record_id"]]

        if (
            row["origin"] != original["label_origin"]
            or row["source"] != original["source"]
            or row["category"] != original["category"]
            or int(row["label"]) != int(row["origin"] == "llm")
            or not row["group_id"]
        ):
            raise ValueError(f"Metadata mismatch: {row['record_id']}")

        group_partitions[row["group_id"]].add(row["partition"])

    if any(len(roles) != 1 for roles in group_partitions.values()):
        raise ValueError("A group crosses partitions.")

    for pair in read_csv(pairs_path):
        if pair["linked"] == "1":
            left = manifest_by_id[pair["left_id"]]
            right = manifest_by_id[pair["right_id"]]
            if left["group_id"] != right["group_id"]:
                raise ValueError("A linked pair has different group IDs.")

    records = sorted(
        (
            row for row in manifest
            if row["partition"] == "development"
            and (
                row["origin"] == "human"
                or row["source"] == generator
            )
        ),
        key=lambda row: row["record_id"],
    )

    forbidden = {
        "id", "text", "label", "label_origin",
        "source", "category", "model",
    }
    if (
        len(FEATURES) != 17
        or len(set(FEATURES)) != 17
        or set(FEATURES) & forbidden
    ):
        raise ValueError("Unexpected feature specification.")

    X = np.asarray(
        [
            [
                float(data_by_id[row["record_id"]][feature])
                for feature in FEATURES
            ]
            for row in records
        ],
        dtype=np.float64,
    )
    y = np.asarray([int(row["label"]) for row in records])
    groups = np.asarray([row["group_id"] for row in records])

    if X.shape != (len(records), 17) or not np.isfinite(X).all():
        raise ValueError("Invalid development feature matrix.")

    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both authorship classes are required.")

    return records, X, y, groups, watched


def prepare_cv(records, y, groups, generator):
    """Load shared folds or prepare new development-only folds."""

    cv_path = SPLITS / f"development_cv_{generator}.csv"
    ids = [row["record_id"] for row in records]

    if cv_path.exists():
        saved = read_csv(cv_path)
        saved_by_id = {row["record_id"]: row for row in saved}

        if (
            len(saved_by_id) != len(saved)
            or set(saved_by_id) != set(ids)
        ):
            raise ValueError("Existing CV manifest has different records.")

        if any(
            saved_by_id[row["record_id"]]["group_id"] != row["group_id"]
            for row in records
        ):
            raise ValueError("Existing CV manifest has different groups.")

        assignments = np.asarray([
            int(saved_by_id[record_id]["validation_fold"])
            for record_id in ids
        ])
    else:
        strata = [
            "|".join((row["origin"], row["source"], row["category"]))
            for row in records
        ]
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        )
        assignments = np.full(len(records), -1, dtype=int)

        for fold, (_, validation) in enumerate(
            splitter.split(
                np.zeros((len(records), 1)),
                strata,
                groups,
            )
        ):
            assignments[validation] = fold

    if set(assignments.tolist()) != set(range(5)):
        raise ValueError("Expected five complete validation folds.")

    folds = []
    maximum_k = max(knn_parameter_grid()["knn__n_neighbors"])

    for fold in range(5):
        training = np.flatnonzero(assignments != fold)
        validation = np.flatnonzero(assignments == fold)

        if set(groups[training]) & set(groups[validation]):
            raise ValueError("Similarity-group leakage inside CV.")

        if (
            set(y[training].tolist()) != {0, 1}
            or set(y[validation].tolist()) != {0, 1}
            or len(training) < maximum_k
        ):
            raise ValueError("A fold cannot support the proposed search.")

        folds.append((training, validation))

    return cv_path, assignments, folds


def f1_at_default_threshold(estimator, X, y):
    """Use one explicit decision rule for parameter selection."""

    scores = predict_llm_scores(estimator, X)
    predicted = (scores >= THRESHOLD).astype(int)

    return f1_score(
        y,
        predicted,
        pos_label=1,
        zero_division=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator",
        choices=["gpt-4.1", "deepseek3.2", "llama-3.3-70b"],
        default="gpt-4.1",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    records, X, y, groups, watched = load_development(args.generator)
    cv_path, assignments, folds = prepare_cv(
        records, y, groups, args.generator
    )

    print("Generator:", args.generator, flush=True)
    print("Development records:", len(records), flush=True)
    print("Class counts:", dict(Counter(y.tolist())), flush=True)
    print("Feature count:", X.shape[1], flush=True)

    for fold, (training, validation) in enumerate(folds):
        print(
            f"Fold {fold}: train={len(training)}, "
            f"validation={len(validation)}, group overlap=0",
            flush=True,
        )

    if args.check_only:
        print("Preflight passed. No model fitted; no files written.")
        return

    if not cv_path.exists():
        write_csv(
            cv_path,
            ["record_id", "group_id", "validation_fold"],
            (
                {
                    "record_id": row["record_id"],
                    "group_id": row["group_id"],
                    "validation_fold": int(assignments[index]),
                }
                for index, row in enumerate(records)
            ),
        )
    watched[cv_path] = sha256(cv_path)
    watched[Path(__file__)] = sha256(Path(__file__))

    search = GridSearchCV(
        estimator=build_knn(),
        param_grid=knn_parameter_grid(),
        scoring={
            "f1": f1_at_default_threshold,
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        },
        refit="f1",
        cv=folds,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
        verbose=2,
    )
    search.fit(X, y)

    print(
        "\nCreating development-only CV prediction records...",
        flush=True,
    )
    # Parameters were selected using these same folds.
    # These predictions are not an unbiased final evaluation.
    probabilities = cross_val_predict(
        search.best_estimator_,
        X,
        y,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    for path, expected_hash in watched.items():
        if sha256(path) != expected_hash:
            raise RuntimeError(f"An input changed during training: {path}")

    run_id = (
        f"knn_{args.generator}_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_"
        f"{uuid4().hex[:8]}"
    )
    output = ROOT / "results" / run_id
    output.mkdir(parents=True, exist_ok=False)

    joblib.dump(search.best_estimator_, output / "model.joblib")

    results = search.cv_results_
    result_keys = sorted(
        key for key in results
        if key.startswith((
            "mean_test_", "std_test_", "rank_test_", "split"
        ))
        or key in {
            "mean_fit_time", "std_fit_time",
            "mean_score_time", "std_score_time",
        }
    )
    write_csv(
        output / "parameter_search.csv",
        ["candidate", "parameters"] + result_keys,
        (
            {
                "candidate": index,
                "parameters": json.dumps(parameters),
                **{key: results[key][index] for key in result_keys},
            }
            for index, parameters in enumerate(results["params"])
        ),
    )

    outcomes = {
        (0, 0): "TN", (0, 1): "FP",
        (1, 0): "FN", (1, 1): "TP",
    }
    write_csv(
        output / "selection_cv_predictions.csv",
        [
            "experiment_id", "record_id", "group_id",
            "validation_fold", "classifier", "training_generator",
            "actual_label", "predicted_label", "llm_score",
            "threshold", "outcome", "explanation_status",
        ] + FEATURES,
        (
            {
                "experiment_id": run_id,
                "record_id": row["record_id"],
                "group_id": row["group_id"],
                "validation_fold": int(assignments[index]),
                "classifier": "KNN",
                "training_generator": args.generator,
                "actual_label": int(y[index]),
                "predicted_label": int(predictions[index]),
                "llm_score": float(probabilities[index]),
                "threshold": THRESHOLD,
                "outcome": outcomes[
                    (int(y[index]), int(predictions[index]))
                ],
                "explanation_status": "not_computed",
                **{
                    feature: float(X[index, column])
                    for column, feature in enumerate(FEATURES)
                },
            }
            for index, row in enumerate(records)
        ),
    )

    best = search.best_index_
    report = {
        "experiment_id": run_id,
        "status": "development_fit_complete",
        "stage": "development_parameter_selection",
        "classifier": "KNN",
        "training_generator": args.generator,
        "development_records": len(records),
        "class_counts": {
            str(label): count
            for label, count in Counter(y.tolist()).items()
        },
        "features": list(FEATURES),
        "feature_version": "original_17_reproduced",
        "split_version": "splits_v2",
        "cv_manifest": str(cv_path),
        "seed": SEED,
        "candidate_count": len(results["params"]),
        "best_parameters": search.best_params_,
        "selection_cv_mean_f1": float(search.best_score_),
        "selection_cv_std_f1": float(results["std_test_f1"][best]),
        "selection_cv_mean_roc_auc": float(
            results["mean_test_roc_auc"][best]
        ),
        "selection_cv_mean_average_precision": float(
            results["mean_test_average_precision"][best]
        ),
        "decision_rule": "LLM if score >= 0.5",
        "score_type": "uncalibrated neighbour-vote estimate",
        "calibration_records_used": 0,
        "final_test_records_used": 0,
        "interpretation": (
            "CV scores and predictions used for parameter selection; "
            "not an unbiased final performance estimate. "
            "Fold standard deviation is not a confidence interval."
        ),
        "versions": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "input_hashes": {
            str(path): value for path, value in watched.items()
        },
        "result_directory": str(output),
    }

    ledger = ROOT / "ledger"
    ledger.mkdir(exist_ok=True)
    report_path = ledger / f"{run_id}.json"

    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))
    print("\nExperiment record:", report_path)
    print("Original inputs unchanged.")
    print("Calibration and final-test records were not used for selection.")


if __name__ == "__main__":
    main()