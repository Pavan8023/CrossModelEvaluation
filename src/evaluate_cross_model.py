"""Cross-generator evaluation of saved development models on held-out test emails."""

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import sklearn

from feature_validation import FEATURES, read_csv, sha256
from train_knn import (
    ROOT, SPLITS, THRESHOLD, load_development, write_csv,
)
from train_lightgbm import check_unchanged
from evaluate_intra_model import (
    CLASSIFIERS, calculate_metrics, llm_scores,
)

GENERATORS = ("gpt-4.1", "deepseek3.2", "llama-3.3-70b")


def load_source(classifier, generator, watched):
    """Verify one development experiment and load its fitted model."""
    matches = []
    folder = ROOT / "ledger" / "development" / classifier

    for path in folder.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("classifier", "").lower() == classifier
            and record.get("training_generator") == generator
            and record.get("status") == "development_fit_complete"
            and record.get("stage") == "development_parameter_selection"
        ):
            matches.append((path, record))

    if len(matches) != 1:
        raise ValueError(
            f"Expected one approved development run for "
            f"{classifier}/{generator}; found {len(matches)}. "
            "Resolve ambiguity before testing."
        )

    ledger_path, record = matches[0]
    rows, X, y, groups, source_inputs = load_development(generator)

    detector_path = ROOT / "src" / f"{classifier}_detector.py"
    cv_path = SPLITS / f"development_cv_{generator}.csv"

    for path in (detector_path, cv_path):
        source_inputs[path] = sha256(path)

    if (
        record["features"] != list(FEATURES)
        or record["feature_version"] != "original_17_reproduced"
        or record["split_version"] != "splits_v2"
        or record["development_records"] != len(rows)
        or record["class_counts"] != {
            str(label): count
            for label, count in Counter(y.tolist()).items()
        }
        or record["calibration_records_used"] != 0
        or record["final_test_records_used"] != 0
    ):
        raise ValueError("Unexpected training provenance.")

    for path, actual in source_inputs.items():
        if record["input_hashes"].get(str(path)) != actual:
            raise ValueError(
                f"Training data, splits or definitions changed: {path}"
            )
        if path in watched and watched[path] != actual:
            raise ValueError(
                f"Input changed while preparing evaluation: {path}"
            )
        watched[path] = actual

    for package in ("scikit_learn", "numpy", "joblib"):
        current = {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        }[package]
        if record["versions"][package] != current:
            raise ValueError(
                f"Training/runtime version differs: {package}"
            )

    if classifier != "knn":
        if record["versions"][classifier] != metadata.version(classifier):
            raise ValueError(
                "Classifier package version differs from training."
            )

    filename = (
        "model.joblib"
        if classifier == "knn"
        else "development_model.joblib"
    )
    model_path = (
        Path(record["result_directory"]) / filename
    ).resolve()

    if not model_path.is_relative_to(ROOT.resolve()):
        raise ValueError("Expected a model inside this project.")

    actual_hash = sha256(model_path)
    historical_hash = record.get("output_hashes", {}).get(filename)

    if historical_hash is not None and historical_hash != actual_hash:
        raise ValueError(
            "Saved model differs from its training artifact hash."
        )
    if classifier != "knn" and historical_hash is None:
        raise ValueError(
            "Expected the model hash in its development ledger."
        )

    model = joblib.load(model_path)
    expected_steps = (
        ["scale", "knn"] if classifier == "knn" else [classifier]
    )
    if list(model.named_steps) != expected_steps:
        raise ValueError("Unexpected saved pipeline steps.")

    # Reloaded CatBoost models retain feature_names_ but may report zero
    # through n_features_in_. Check the retained feature names instead.
    feature_count = (
        len(model.named_steps["catboost"].feature_names_)
        if classifier == "catboost"
        else model.n_features_in_
    )
    if list(model.classes_) != [0, 1] or feature_count != len(FEATURES):
        raise ValueError("Unexpected saved classes or feature count.")

    parameters = model.get_params()
    for name, value in record["best_parameters"].items():
        if parameters.get(name) != value:
            raise ValueError(f"Saved model settings differ: {name}")

    # Early KNN ledgers did not record output hashes.
    # Check stored training examples, labels and scaler statistics.
    if classifier == "knn":
        scaler = model.named_steps["scale"]
        knn = model.named_steps["knn"]

        if (
            knn.n_samples_fit_ != len(rows)
            or not np.allclose(scaler.mean_, X.mean(axis=0))
            or not np.allclose(scaler.var_, X.var(axis=0))
            or not np.allclose(knn._fit_X, scaler.transform(X))
            or not np.array_equal(knn._y, y)
        ):
            raise ValueError(
                "KNN fitted training data or scaler differs."
            )

    watched[ledger_path] = sha256(ledger_path)
    watched[model_path] = actual_hash

    provenance = {
        "experiment_id": record["experiment_id"],
        "ledger_path": str(ledger_path),
        "model_path": str(model_path),
        "model_sha256": actual_hash,
        "historical_model_hash_available": historical_hash is not None,
        "knn_fitted_training_arrays_checked": classifier == "knn",
        "feature_count_checked_via": (
            "feature_names_"
            if classifier == "catboost"
            else "n_features_in_"
        ),
        "development_records": len(rows),
        "class_counts": record["class_counts"],
        "selected_parameters": record["best_parameters"],
        "training_versions": record["versions"],
    }

    return {
        "model": model,
        "training_ids": {row["record_id"] for row in rows},
        "training_groups": set(groups.tolist()),
        "provenance": provenance,
    }


def prepare_test_sets():
    """Build fixed target sets from the reserved test partition."""
    data = {
        row["id"]: row
        for row in read_csv(ROOT / "data" / "corpus_features.csv")
    }
    manifest = read_csv(SPLITS / "manifest.csv")
    targets = {}

    for generator in GENERATORS:
        rows = sorted(
            (
                row for row in manifest
                if row["partition"] == "test"
                and (
                    row["origin"] == "human"
                    or row["source"] == generator
                )
            ),
            key=lambda row: row["record_id"],
        )

        X = np.asarray([
            [
                float(data[row["record_id"]][feature])
                for feature in FEATURES
            ]
            for row in rows
        ], dtype=np.float64)

        y = np.asarray(
            [int(row["label"]) for row in rows], dtype=int
        )

        if (
            X.shape != (len(rows), len(FEATURES))
            or not np.isfinite(X).all()
            or set(y.tolist()) != {0, 1}
        ):
            raise ValueError("Invalid held-out target data.")

        targets[generator] = {
            "rows": rows,
            "X": X,
            "y": y,
            "ids": {row["record_id"] for row in rows},
            "groups": {row["group_id"] for row in rows},
        }

    return targets


def gap_summary(cells):
    """Use equal-weight matrix cells, not nested-CV scores."""
    same = [
        row["f1"] for row in cells
        if row["training_generator"] == row["evaluation_generator"]
    ]
    different = [
        row["f1"] for row in cells
        if row["training_generator"] != row["evaluation_generator"]
    ]

    if len(same) != 3 or len(different) != 6:
        raise ValueError("A complete 3-by-3 matrix is required.")

    diagonal = float(np.mean(same))
    off_diagonal = float(np.mean(different))

    return {
        "mean_same_generator_f1": diagonal,
        "mean_different_generator_f1": off_diagonal,
        "performance_gap_f1": diagonal - off_diagonal,
        "performance_gap_percentage_points": (
            100 * (diagonal - off_diagonal)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier", choices=CLASSIFIERS, required=True
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if THRESHOLD != 0.5:
        raise ValueError(
            "Protocol v1 requires the default threshold 0.5."
        )

    watched = {}
    sources = {
        generator: load_source(
            args.classifier, generator, watched
        )
        for generator in GENERATORS
    }
    targets = prepare_test_sets()

    for source in sources.values():
        for target in targets.values():
            if (
                source["training_ids"] & target["ids"]
                or source["training_groups"] & target["groups"]
            ):
                raise ValueError(
                    "Training/test record or group overlap."
                )

    for path in (
        ROOT / "src" / "train_knn.py",
        ROOT / "src" / "train_lightgbm.py",
        ROOT / "src" / "evaluate_intra_model.py",
    ):
        watched[path] = sha256(path)

    print("Evaluation: cross-model held-out test", flush=True)
    print("Classifier:", args.classifier, flush=True)
    print(
        "Saved development models verified:", len(sources),
        flush=True,
    )

    for generator, target in targets.items():
        print(
            f"Target {generator}: {len(target['rows'])} test records; "
            f"classes={dict(Counter(target['y'].tolist()))}",
            flush=True,
        )

    print(
        "All nine training/test group checks passed.",
        flush=True,
    )
    print(
        "Threshold:", THRESHOLD,
        "; planned model fits: 0",
        flush=True,
    )

    check_unchanged(watched)
    if args.check_only:
        print(
            "Preflight passed. No test predictions; no files written."
        )
        return

    watched[Path(__file__).resolve()] = sha256(
        Path(__file__).resolve()
    )

    run_id = (
        f"{args.classifier}_cross_model_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_"
        f"{uuid4().hex[:8]}"
    )
    cells, predictions = [], []
    outcomes = {
        (0, 0): "TN", (0, 1): "FP",
        (1, 0): "FN", (1, 1): "TP",
    }

    for training_generator, source in sources.items():
        for evaluation_generator, target in targets.items():
            scores = llm_scores(source["model"], target["X"])
            predicted = (scores >= THRESHOLD).astype(int)

            cells.append({
                "training_generator": training_generator,
                "evaluation_generator": evaluation_generator,
                "condition": (
                    "same_generator"
                    if training_generator == evaluation_generator
                    else "different_generator"
                ),
                "test_records": len(target["rows"]),
                "human_test_records": int(
                    np.sum(target["y"] == 0)
                ),
                "llm_test_records": int(
                    np.sum(target["y"] == 1)
                ),
                "threshold": THRESHOLD,
                **calculate_metrics(target["y"], scores),
            })

            for index, row in enumerate(target["rows"]):
                actual = int(target["y"][index])
                prediction = int(predicted[index])

                predictions.append({
                    "experiment_id": run_id,
                    "record_id": row["record_id"],
                    "group_id": row["group_id"],
                    "classifier": args.classifier,
                    "training_generator": training_generator,
                    "evaluation_generator": evaluation_generator,
                    "email_source": row["source"],
                    "source_experiment_id": (
                        source["provenance"]["experiment_id"]
                    ),
                    "actual_label": actual,
                    "predicted_label": prediction,
                    "llm_score": float(scores[index]),
                    "threshold": THRESHOLD,
                    "outcome": outcomes[(actual, prediction)],
                    "correct": actual == prediction,
                    "explanation_status": "not_computed",
                    **{
                        feature: float(target["X"][index, column])
                        for column, feature in enumerate(FEATURES)
                    },
                })

    check_unchanged(watched)

    diagonal = {
        row["training_generator"]: row["f1"]
        for row in cells
        if row["training_generator"] == row["evaluation_generator"]
    }
    for row in cells:
        row["source_diagonal_f1_minus_cell_f1"] = (
            diagonal[row["training_generator"]] - row["f1"]
        )

    output = (
        ROOT / "results" / "cross_model"
        / args.classifier / run_id
    )
    output.mkdir(parents=True, exist_ok=False)

    write_csv(
        output / "cross_model_metrics.csv",
        list(cells[0]), cells,
    )
    write_csv(
        output / "cross_model_predictions.csv",
        list(predictions[0]), predictions,
    )

    matrix = [
        {
            "training_generator": source,
            **{
                target: next(
                    row["f1"] for row in cells
                    if row["training_generator"] == source
                    and row["evaluation_generator"] == target
                )
                for target in GENERATORS
            },
        }
        for source in GENERATORS
    ]

    write_csv(
        output / "cross_model_f1_matrix.csv",
        ["training_generator"] + list(GENERATORS),
        matrix,
    )

    gap = gap_summary(cells)

    report = {
        "experiment_id": run_id,
        "status": "cross_model_complete",
        "evaluation_method": "cross_model",
        "classifier": args.classifier,
        "protocol_version": "cross_model_v1",
        "features": list(FEATURES),
        "feature_version": "original_17_reproduced",
        "split_version": "splits_v2",
        "evaluation_partition": "test",
        "generators": list(GENERATORS),
        "source_models": {
            source: value["provenance"]
            for source, value in sources.items()
        },
        "decision_rule": f"LLM if score >= {THRESHOLD}",
        "positive_class": "LLM",
        "fit_count": 0,
        "cell_count": len(cells),
        "cell_metrics": cells,
        **gap,
        "gap_definition": (
            "Equal-weight mean of three held-out same-generator "
            "F1 cells minus equal-weight mean of six held-out "
            "different-generator F1 cells."
        ),
        "calibration_records_used": 0,
        "final_test_records_used": len(set().union(
            *(target["ids"] for target in targets.values())
        )),
        "prediction_rows": len(predictions),
        "interpretation": (
            "Frozen development models; no fitting, threshold "
            "selection or hyperparameter selection on test records. "
            "The same held-out human records appear in all cells. "
            "Email IDs therefore repeat across different generator "
            "pairs; cells are dependent. No combined metric over all "
            "prediction rows is reported. Original class proportions "
            "are retained. Nested-CV F1 values are not used in this gap."
        ),
        "uncertainty": (
            "Not estimated in this step; retain groups for later analysis."
        ),
        "planned_threshold_recalibration": (
            "Use target calibration records only; maximize calibration "
            "F1; ties prefer the threshold closest to 0.5, then the "
            "larger threshold."
        ),
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

    check_unchanged(watched)

    ledger = (
        ROOT / "ledger" / "cross_model" / args.classifier
    )
    ledger.mkdir(parents=True, exist_ok=True)
    report_path = ledger / f"{run_id}.json"

    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(
        "F1 matrix: training generator = rows; "
        "test generator = columns"
    )
    for row in matrix:
        print(row)

    print("Gap summary:", gap)
    print("Experiment record:", report_path)
    print("No model fitted. Calibration records were not used.")


if __name__ == "__main__":
    main()