"""Nested, group-aware intra-model evaluation on development records only."""

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedGroupKFold,
)

from feature_validation import FEATURES, read_csv, sha256
from train_knn import (
    ROOT, SPLITS, SEED, THRESHOLD,
    load_development, prepare_cv, write_csv,
)
from train_lightgbm import check_knn_reference, check_unchanged

CLASSIFIERS = ("knn", "lightgbm", "catboost")
N_INNER = 5


def llm_scores(model, X):
    """Read class-1 scores and validate their meaning."""
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected classes 0=human and 1=LLM.")

    scores = np.asarray(model.predict_proba(X)[:, 1], dtype=float)

    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid probability scores.")

    return scores


def selection_f1(model, X, y):
    predicted = (llm_scores(model, X) >= THRESHOLD).astype(int)
    return metrics.f1_score(
        y, predicted, pos_label=1, zero_division=0
    )


def calculate_metrics(y, scores):
    predicted = (scores >= THRESHOLD).astype(int)

    tn, fp, fn, tp = metrics.confusion_matrix(
        y, predicted, labels=[0, 1]
    ).ravel()

    recall = float(
        metrics.recall_score(y, predicted, zero_division=0)
    )

    return {
        "accuracy": float(metrics.accuracy_score(y, predicted)),
        "precision": float(
            metrics.precision_score(y, predicted, zero_division=0)
        ),
        "recall": recall,
        "sensitivity": recall,
        "specificity": float(tn / (tn + fp)),
        "f1": float(
            metrics.f1_score(y, predicted, zero_division=0)
        ),
        "roc_auc": float(metrics.roc_auc_score(y, scores)),
        "average_precision": float(
            metrics.average_precision_score(y, scores)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_nested_plan(records, y, groups, outer_folds):
    """Create classifier-independent inner folds and an audit manifest."""
    strata = np.asarray([
        "|".join((row["origin"], row["source"], row["category"]))
        for row in records
    ])

    plan, manifest = [], []

    for outer_fold, (training, evaluation) in enumerate(outer_folds):
        if set(groups[training]) & set(groups[evaluation]):
            raise ValueError("Outer group leakage.")

        splitter = StratifiedGroupKFold(
            n_splits=N_INNER,
            shuffle=True,
            random_state=SEED + outer_fold + 1,
        )

        inner_folds = list(splitter.split(
            np.zeros((len(training), 1)),
            strata[training],
            groups[training],
        ))

        inner_assignments = np.full(len(records), -1, dtype=int)
        coverage = np.zeros(len(training), dtype=int)

        for inner_fold, (fit, validate) in enumerate(inner_folds):
            fit_global = training[fit]
            validate_global = training[validate]

            if set(groups[fit_global]) & set(groups[validate_global]):
                raise ValueError("Inner group leakage.")

            if (
                set(y[fit_global].tolist()) != {0, 1}
                or set(y[validate_global].tolist()) != {0, 1}
                or len(fit) < 31
            ):
                raise ValueError(
                    "Inner fold cannot support the approved grids."
                )

            inner_assignments[validate_global] = inner_fold
            coverage[validate] += 1

        if not np.all(coverage == 1):
            raise ValueError(
                "Incomplete or repeated inner validation coverage."
            )

        if not np.all(inner_assignments[evaluation] == -1):
            raise ValueError(
                "Outer evaluation records entered inner selection."
            )

        plan.append((training, evaluation, inner_folds))

        for index, row in enumerate(records):
            inner_fold = int(inner_assignments[index])
            manifest.append({
                "outer_fold": str(outer_fold),
                "record_id": row["record_id"],
                "group_id": row["group_id"],
                "role": (
                    "outer_evaluation"
                    if inner_fold == -1 else "outer_training"
                ),
                "inner_validation_fold": str(inner_fold),
            })

    return plan, manifest


def save_search(path, results):
    """Save inner-fold search scores."""
    keys = sorted(
        key for key in results
        if key.startswith((
            "mean_test_", "std_test_", "rank_test_", "split"
        ))
        or key in {"mean_fit_time", "mean_score_time"}
    )

    write_csv(
        path,
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--classifier", choices=CLASSIFIERS, required=True
    )
    parser.add_argument(
        "--generator",
        choices=["gpt-4.1", "deepseek3.2", "llama-3.3-70b"],
        required=True,
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    module = import_module(f"{args.classifier}_detector")
    factory = getattr(module, f"build_{args.classifier}")
    grid = getattr(
        module, f"{args.classifier}_parameter_grid"
    )()

    if len(ParameterGrid(grid)) != 20:
        raise ValueError("Expected the approved 20-candidate search.")

    cv_path = SPLITS / f"development_cv_{args.generator}.csv"
    if not cv_path.is_file():
        raise FileNotFoundError(
            "The shared development CV manifest is required."
        )

    records, X, y, groups, watched = load_development(
        args.generator
    )
    watched[cv_path] = sha256(cv_path)

    _, outer_assignments, outer_folds = prepare_cv(
        records, y, groups, args.generator
    )
    reference_ids = check_knn_reference(args.generator, watched)

    plan, manifest = make_nested_plan(
        records, y, groups, outer_folds
    )

    nested_path = SPLITS / f"nested_cv_v1_{args.generator}.csv"

    if nested_path.exists():
        watched[nested_path] = sha256(nested_path)
        if read_csv(nested_path) != manifest:
            raise ValueError(
                "Saved nested plan differs from protocol v1."
            )

    for path in (
        Path(module.__file__).resolve(),
        ROOT / "src" / "train_knn.py",
        ROOT / "src" / "train_lightgbm.py",
    ):
        watched[path] = sha256(path)

    print("Evaluation: intra-model nested CV", flush=True)
    print("Classifier:", args.classifier, flush=True)
    print("Generator:", args.generator, flush=True)
    print("Development records:", len(records), flush=True)
    print("Features:", X.shape[1], flush=True)
    print(
        "Outer folds: 5; inner folds: 5; candidates: 20",
        flush=True,
    )
    print(
        "Planned fits: 505, including five outer-training refits",
        flush=True,
    )

    for fold, (training, evaluation, inner) in enumerate(plan):
        print(
            f"Outer {fold}: training={len(training)}, "
            f"evaluation={len(evaluation)}, "
            f"smallest inner training="
            f"{min(len(a) for a, _ in inner)}; "
            "group checks passed",
            flush=True,
        )

    check_unchanged(watched)

    if args.check_only:
        print("Preflight passed. No model fitted; no files written.")
        return

    watched[Path(__file__).resolve()] = sha256(
        Path(__file__).resolve()
    )

    if not nested_path.exists():
        write_csv(nested_path, list(manifest[0]), manifest)
    watched[nested_path] = sha256(nested_path)

    run_id = (
        f"{args.classifier}_intra_model_{args.generator}_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_"
        f"{uuid4().hex[:8]}"
    )

    output = (
        ROOT / "results" / "intra_model"
        / args.classifier / run_id
    )
    output.mkdir(parents=True, exist_ok=False)

    oof_scores = np.full(len(records), np.nan)
    coverage = np.zeros(len(records), dtype=int)
    fold_records, fold_configurations = [], []

    for fold, (training, evaluation, inner_folds) in enumerate(plan):
        check_unchanged(watched)
        print(f"Starting outer fold {fold + 1}/5", flush=True)

        search = GridSearchCV(
            estimator=factory(),
            param_grid=grid,
            scoring={
                "f1": selection_f1,
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
            },
            refit="f1",
            cv=inner_folds,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
            verbose=1,
        )

        # Inner indices refer to this outer-training subset.
        search.fit(X[training], y[training])

        scores = llm_scores(
            search.best_estimator_, X[evaluation]
        )
        oof_scores[evaluation] = scores
        coverage[evaluation] += 1
        check_unchanged(watched)

        fold_records.append({
            "outer_fold": fold,
            "training_records": len(training),
            "evaluation_records": len(evaluation),
            "inner_selection_mean_f1": float(search.best_score_),
            "best_parameters": json.dumps(search.best_params_),
            **calculate_metrics(y[evaluation], scores),
        })

        fitted = search.best_estimator_.named_steps[
            args.classifier
        ]
        parameters = (
            fitted.get_all_params()
            if args.classifier == "catboost"
            else fitted.get_params()
        )

        fold_configurations.append({
            "outer_fold": fold,
            "selected_parameters": search.best_params_,
            "fitted_estimator_parameters": parameters,
        })

        joblib.dump(
            search.best_estimator_,
            output / f"outer_fold_{fold}_model.joblib",
        )
        save_search(
            output / f"outer_fold_{fold}_inner_search.csv",
            search.cv_results_,
        )

    if (
        not np.all(coverage == 1)
        or not np.isfinite(oof_scores).all()
    ):
        raise RuntimeError(
            "Each record must receive exactly one outer prediction."
        )

    write_csv(
        output / "intra_model_fold_metrics.csv",
        list(fold_records[0]),
        fold_records,
    )

    predicted = (oof_scores >= THRESHOLD).astype(int)
    outcomes = {
        (0, 0): "TN", (0, 1): "FP",
        (1, 0): "FN", (1, 1): "TP",
    }

    fields = [
        "experiment_id", "record_id", "group_id", "outer_fold",
        "classifier", "training_generator", "evaluation_generator",
        "email_source", "actual_label", "predicted_label", "llm_score",
        "threshold", "outcome", "correct", "model_file",
        "explanation_status",
    ] + list(FEATURES)

    write_csv(
        output / "intra_model_predictions.csv",
        fields,
        (
            {
                "experiment_id": run_id,
                "record_id": row["record_id"],
                "group_id": row["group_id"],
                "outer_fold": int(outer_assignments[index]),
                "classifier": args.classifier,
                "training_generator": args.generator,
                "evaluation_generator": args.generator,
                "email_source": row["source"],
                "actual_label": int(y[index]),
                "predicted_label": int(predicted[index]),
                "llm_score": float(oof_scores[index]),
                "threshold": THRESHOLD,
                "outcome": outcomes[
                    (int(y[index]), int(predicted[index]))
                ],
                "correct": bool(y[index] == predicted[index]),
                "model_file": (
                    f"outer_fold_{int(outer_assignments[index])}"
                    "_model.joblib"
                ),
                "explanation_status": "not_computed",
                **{
                    feature: float(X[index, column])
                    for column, feature in enumerate(FEATURES)
                },
            }
            for index, row in enumerate(records)
        ),
    )

    metric_names = [
        "accuracy", "precision", "recall", "sensitivity",
        "specificity", "f1", "roc_auc", "average_precision",
    ]

    outer_summary = {
        name: {
            "mean": float(np.mean([
                row[name] for row in fold_records
            ])),
            "std": float(np.std([
                row[name] for row in fold_records
            ], ddof=1)),
        }
        for name in metric_names
    }

    check_unchanged(watched)

    report = {
        "experiment_id": run_id,
        "status": "intra_model_nested_cv_complete",
        "evaluation_method": "intra_model",
        "classifier": args.classifier,
        "training_generator": args.generator,
        "evaluation_generator": args.generator,
        "evaluation_partition": "development_outer_folds",
        "development_records": len(records),
        "features": list(FEATURES),
        "feature_version": "original_17_reproduced",
        "split_version": "splits_v2",
        "nested_protocol": (
            "v1: saved outer folds; five inner group folds"
        ),
        "outer_cv_manifest": str(cv_path),
        "nested_cv_manifest": str(nested_path),
        "knn_reference_experiments": reference_ids,
        "outer_folds": 5,
        "inner_folds": N_INNER,
        "outer_seed": SEED,
        "inner_seeds": [
            SEED + fold + 1 for fold in range(5)
        ],
        "parameter_grid": grid,
        "candidate_count": len(ParameterGrid(grid)),
        "fit_count": 505,
        "decision_rule": f"LLM if score >= {THRESHOLD}",
        "positive_class": "LLM",
        "outer_fold_metrics": outer_summary,
        "pooled_outer_predictions_metrics": calculate_metrics(
            y, oof_scores
        ),
        "fold_configurations": fold_configurations,
        "uncertainty": (
            "Sample standard deviation across five outer folds "
            "(ddof=1); not a confidence interval; folds have "
            "overlapping training data."
        ),
        "interpretation": (
            "Each outer prediction excludes that record and its "
            "similarity group from its model's fitting and inner "
            "parameter selection. Development data were explored "
            "previously; this is internal nested evaluation, not "
            "an untouched final-test result. Do not change the "
            "grids using outer evaluation scores."
        ),
        "calibration_records_used": 0,
        "final_test_records_used": 0,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            args.classifier: (
                sklearn.__version__
                if args.classifier == "knn"
                else metadata.version(args.classifier)
            ),
        },
        "input_hashes": {
            str(path): value for path, value in watched.items()
        },
        "result_directory": str(output),
        "output_hashes": {
            path.name: sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }

    ledger = (
        ROOT / "ledger" / "intra_model" / args.classifier
    )
    ledger.mkdir(parents=True, exist_ok=True)
    report_path = ledger / f"{run_id}.json"

    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print("Mean outer-fold F1:", outer_summary["f1"]["mean"])
    print(
        "Pooled outer prediction metrics:",
        report["pooled_outer_predictions_metrics"],
    )
    print("Experiment record:", report_path)
    print("Calibration and final-test records were not used.")


if __name__ == "__main__":
    main()