"""Development-only LightGBM selection using the shared KNN partitions."""

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_predict

from feature_validation import FEATURES, sha256
from lightgbm_detector import (
    build_lightgbm,
    lightgbm_parameter_grid,
    predict_llm_scores,
)
from train_knn import (
    ROOT, SPLITS, SEED, THRESHOLD,
    load_development, prepare_cv, write_csv,
)


def check_unchanged(watched):
    """Stop if a tracked input has changed."""
    for path, expected in watched.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Input changed: {path}")


def check_knn_reference(generator, watched):
    """Confirm that shared inputs match the completed KNN run."""
    reference_ids = []
    shared_paths = [
        path for path in watched
        if path.parent == ROOT / "data"
        or path.parent == SPLITS
        or path.name == "feature_validation.py"
    ]

    reference_folder = ROOT / "ledger" / "development" / "knn"
    for path in sorted(reference_folder.glob("*.json")):
        reference = json.loads(path.read_text(encoding="utf-8"))
        if (
            reference.get("classifier") != "KNN"
            or reference.get("training_generator") != generator
            or reference.get("stage") != "development_parameter_selection"
            or reference.get("status") != "development_fit_complete"
        ):
            continue

        if reference["features"] != list(FEATURES):
            raise ValueError("Feature list differs from the KNN reference.")

        for shared_path in shared_paths:
            expected = reference["input_hashes"].get(str(shared_path))
            if expected != watched[shared_path]:
                raise ValueError(f"Shared input differs from KNN: {shared_path}")

        reference_ids.append(reference["experiment_id"])
        watched[path] = sha256(path)

    if not reference_ids:
        raise ValueError("No completed KNN reference found for this generator.")
    return reference_ids


def f1_at_default_threshold(estimator, X, y):
    """Select settings using the same threshold rule as KNN."""
    predicted = (predict_llm_scores(estimator, X) >= THRESHOLD).astype(int)
    return f1_score(y, predicted, pos_label=1, zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator",
        choices=["gpt-4.1", "deepseek3.2", "llama-3.3-70b"],
        default="gpt-4.1",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    cv_path = SPLITS / f"development_cv_{args.generator}.csv"
    if not cv_path.is_file():
        raise FileNotFoundError("The saved shared KNN CV manifest is required.")

    records, X, y, groups, watched = load_development(args.generator)
    watched[cv_path] = sha256(cv_path)
    cv_path, assignments, folds = prepare_cv(records, y, groups, args.generator)
    reference_ids = check_knn_reference(args.generator, watched)

    for path in (
        ROOT / "src" / "lightgbm_detector.py",
        ROOT / "src" / "train_knn.py",
    ):
        watched[path] = sha256(path)

    grid = lightgbm_parameter_grid()
    print("LightGBM:", lgb.__version__, flush=True)
    print("Generator:", args.generator, flush=True)
    print("Development records:", len(records), flush=True)
    print("Class counts:", dict(Counter(y.tolist())), flush=True)
    print("Features:", X.shape[1], flush=True)
    print("Parameter combinations:", len(ParameterGrid(grid)), flush=True)
    print("Shared inputs match KNN:", reference_ids, flush=True)

    for fold, (training, validation) in enumerate(folds):
        print(
            f"Fold {fold}: train={len(training)}, "
            f"validation={len(validation)}, group overlap=0",
            flush=True,
        )

    check_unchanged(watched)
    if args.check_only:
        print("Preflight passed. No model fitted; no files written.")
        return

    watched[Path(__file__).resolve()] = sha256(Path(__file__).resolve())

    search = GridSearchCV(
        estimator=build_lightgbm(),
        param_grid=grid,
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

    # These folds also selected the settings: this is development evidence,
    # not an unbiased final evaluation.
    probabilities = cross_val_predict(
        search.best_estimator_, X, y,
        cv=folds, method="predict_proba", n_jobs=1,
    )[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    check_unchanged(watched)

    run_id = (
        f"lightgbm_development_{args.generator}_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_"
        f"{uuid4().hex[:8]}"
    )
    output = ROOT / "results" / "development" / "lightgbm" / run_id
    output.mkdir(parents=True, exist_ok=False)

    joblib.dump(
        search.best_estimator_, output / "development_model.joblib"
    )

    results = search.cv_results_
    keys = sorted(
        key for key in results
        if key.startswith(("mean_test_", "std_test_", "rank_test_", "split"))
        or key in {"mean_fit_time", "mean_score_time"}
    )
    write_csv(
        output / "development_parameter_search.csv",
        ["candidate", "parameters"] + keys,
        (
            {
                "candidate": index,
                "parameters": json.dumps(parameters),
                **{key: results[key][index] for key in keys},
            }
            for index, parameters in enumerate(results["params"])
        ),
    )

    outcomes = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
    fields = [
        "experiment_id", "record_id", "group_id", "validation_fold",
        "classifier", "training_generator", "evaluation_generator",
        "email_source", "actual_label", "predicted_label", "llm_score",
        "threshold", "outcome", "explanation_status",
    ] + list(FEATURES)

    write_csv(
        output / "development_selection_cv_predictions.csv",
        fields,
        (
            {
                "experiment_id": run_id,
                "record_id": row["record_id"],
                "group_id": row["group_id"],
                "validation_fold": int(assignments[index]),
                "classifier": "LightGBM",
                "training_generator": args.generator,
                "evaluation_generator": args.generator,
                "email_source": row["source"],
                "actual_label": int(y[index]),
                "predicted_label": int(predictions[index]),
                "llm_score": float(probabilities[index]),
                "threshold": THRESHOLD,
                "outcome": outcomes[(int(y[index]), int(predictions[index]))],
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
        "classifier": "LightGBM",
        "training_generator": args.generator,
        "development_records": len(records),
        "class_counts": {
            str(label): count for label, count in Counter(y.tolist()).items()
        },
        "features": list(FEATURES),
        "feature_version": "original_17_reproduced",
        "split_version": "splits_v2",
        "cv_manifest": str(cv_path),
        "knn_reference_experiments": reference_ids,
        "seed": SEED,
        "candidate_count": len(results["params"]),
        "best_parameters": search.best_params_,
        "full_model_parameters": (
            search.best_estimator_.named_steps["lightgbm"].get_params()
        ),
        "selection_cv_mean_f1": float(search.best_score_),
        "selection_cv_std_f1": float(results["std_test_f1"][best]),
        "selection_cv_mean_roc_auc": float(results["mean_test_roc_auc"][best]),
        "selection_cv_mean_average_precision": float(
            results["mean_test_average_precision"][best]
        ),
        "decision_rule": f"LLM if score >= {THRESHOLD}",
        "score_type": "uncalibrated LightGBM class-1 probability estimate",
        "preprocessing": "original numerical features; no standardization",
        "calibration_records_used": 0,
        "final_test_records_used": 0,
        "interpretation": (
            "CV scores and predictions used for parameter selection; "
            "not an unbiased final performance estimate. "
            "Fold standard deviation is not a confidence interval."
        ),
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "input_hashes": {str(path): value for path, value in watched.items()},
        "result_directory": str(output),
        "output_hashes": {
            path.name: sha256(path) for path in sorted(output.iterdir())
            if path.is_file()
        },
    }

    ledger = ROOT / "ledger" / "development" / "lightgbm"
    ledger.mkdir(parents=True, exist_ok=True)
    report_path = ledger / f"{run_id}.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))
    print("Experiment record:", report_path)
    print("Original inputs unchanged.")
    print("Calibration and final-test records were not used for selection.")


if __name__ == "__main__":
    main()