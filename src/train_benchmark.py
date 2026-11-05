"""Fit an additional benchmark classifier on development data only."""

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, ParameterGrid

from evaluate_cross_model import GENERATORS
from evaluate_intra_model import save_search, selection_f1
from feature_validation import FEATURES, sha256
from train_knn import ROOT, SEED, SPLITS, load_development, prepare_cv
from train_lightgbm import check_knn_reference, check_unchanged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier", choices=("xgboost", "random_forest"), required=True
    )
    parser.add_argument("--generator", choices=GENERATORS, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    key = args.classifier
    detector = import_module(f"{key}_detector")
    factory = getattr(detector, f"build_{key}")
    grid = getattr(detector, f"{key}_parameter_grid")()
    candidates = len(ParameterGrid(grid))
    if candidates != 20:
        raise ValueError("Expected 20 benchmark settings")

    cv_file = SPLITS / f"development_cv_{args.generator}.csv"
    if not cv_file.is_file():
        raise FileNotFoundError(f"Shared folds are required: {cv_file}")

    records, X, y, groups, watched = load_development(args.generator)
    watched[cv_file] = sha256(cv_file)
    cv_path, _, folds = prepare_cv(records, y, groups, args.generator)
    references = check_knn_reference(args.generator, watched)

    for path in (
        Path(detector.__file__).resolve(),
        Path(__file__).resolve(),
        ROOT / "src" / "train_knn.py",
        ROOT / "src" / "train_lightgbm.py",
        ROOT / "src" / "evaluate_intra_model.py",
    ):
        watched[path] = sha256(path)

    check_unchanged(watched)
    print(
        f"{key} / {args.generator}: {len(records)} records, "
        f"{X.shape[1]} features, {len(folds)} group folds, "
        f"{candidates} settings",
        flush=True,
    )
    if args.check_only:
        print("Preflight passed.")
        return

    ledger_dir = ROOT / "ledger" / "development" / key
    if any(ledger_dir.glob(f"{key}_development_{args.generator}_*.json")):
        raise FileExistsError(
            "A development ledger already exists for this classifier and generator"
        )

    search = GridSearchCV(
        factory(),
        grid,
        cv=folds,
        scoring={
            "f1": selection_f1,
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        },
        refit="f1",
        n_jobs=1,
        error_score="raise",
        verbose=1,
    )
    search.fit(X, y)
    check_unchanged(watched)

    run_id = (
        f"{key}_development_{args.generator}_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    )
    output = ROOT / "results" / "development" / key / run_id
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(search.best_estimator_, output / "development_model.joblib")
    save_search(output / "development_parameter_search.csv", search.cv_results_)

    package = "scikit-learn" if key == "random_forest" else key
    parameters = search.best_estimator_.named_steps[key].get_params()
    recorded_parameters = {
        name: str(value)
        if isinstance(value, float) and not np.isfinite(value)
        else value
        for name, value in parameters.items()
    }
    report = {
        "experiment_id": run_id,
        "status": "development_fit_complete",
        "stage": "development_parameter_selection",
        "classifier": key,
        "training_generator": args.generator,
        "development_records": len(records),
        "class_counts": {str(k): v for k, v in Counter(y.tolist()).items()},
        "features": list(FEATURES),
        "feature_version": "original_17_reproduced",
        "split_version": "splits_v2",
        "cv_manifest": str(cv_path),
        "knn_reference_experiments": references,
        "seed": SEED,
        "candidate_count": candidates,
        "fit_count": candidates * len(folds) + 1,
        "best_parameters": search.best_params_,
        "full_model_parameters": recorded_parameters,
        "selection_cv_mean_f1": float(search.best_score_),
        "calibration_records_used": 0,
        "final_test_records_used": 0,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            key: version(package),
        },
        "input_hashes": {str(path): value for path, value in watched.items()},
        "result_directory": str(output),
        "output_hashes": {p.name: sha256(p) for p in output.iterdir()},
    }
    check_unchanged(watched)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / f"{run_id}.json"
    with ledger_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print("Development ledger:", ledger_path)


if __name__ == "__main__":
    main()