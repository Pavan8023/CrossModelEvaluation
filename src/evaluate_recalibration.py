"""Tune thresholds on calibration data and evaluate saved test scores."""

import argparse
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
from itertools import product
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

from evaluate_cross_model import CLASSIFIERS, GENERATORS, gap_summary
from evaluate_intra_model import llm_scores
from feature_validation import FEATURES, read_csv, sha256
from train_knn import ROOT, SPLITS, THRESHOLD, write_csv
from train_lightgbm import check_unchanged

PAIRS = tuple(product(GENERATORS, repeat=2))
OUTCOME = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
RULE = (
    "Use target calibration records only; maximize calibration F1; "
    "ties prefer the threshold closest to 0.5, then the larger threshold."
)


def choose_threshold(y, scores):
    """Search observed calibration scores plus the default threshold."""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    positives = int(y.sum())
    prefix = np.r_[0, np.cumsum(y[order])]
    candidates = np.unique(np.r_[scores, THRESHOLD])
    ranked = []

    for threshold, left in zip(
        candidates, np.searchsorted(sorted_scores, candidates, side="left")
    ):
        tp = positives - int(prefix[left])
        predicted_positive = len(y) - int(left)
        f1 = Fraction(2 * tp, positives + predicted_positive)
        t = float(threshold)
        ranked.append((f1, -abs(t - THRESHOLD), t))

    f1, _, threshold = max(ranked)
    return threshold, float(f1), len(candidates)


def label_metrics(y, predicted):
    tn, fp, fn, tp = map(
        int, confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    )
    recall = tp / (tp + fn)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tn + tp) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": recall,
        "sensitivity": recall,
        "specificity": tn / (tn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def load_baseline(classifier):
    folder = ROOT / "ledger" / "cross_model" / classifier
    matches = []
    for path in folder.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "cross_model_complete"
            and report.get("classifier") == classifier
        ):
            matches.append((path, report))

    if len(matches) != 1:
        raise ValueError(f"Expected one completed cross-model ledger in {folder}.")

    path, report = matches[0]
    if (
        report["protocol_version"] != "cross_model_v1"
        or report["planned_threshold_recalibration"] != RULE
        or report["features"] != list(FEATURES)
        or report["decision_rule"] != "LLM if score >= 0.5"
        or THRESHOLD != 0.5
    ):
        raise ValueError("Unexpected baseline protocol.")

    packages = {"scikit_learn": "scikit-learn", "numpy": "numpy", "joblib": "joblib"}
    if classifier != "knn":
        packages[classifier] = classifier
    current_versions = {key: version(package) for key, package in packages.items()}
    current_versions["python"] = platform.python_version()
    if any(report["versions"][key] != value for key, value in current_versions.items()):
        raise ValueError("Runtime versions differ from the cross-model run.")

    watched = {Path(name): value for name, value in report["input_hashes"].items()}
    watched[path] = sha256(path)
    output = Path(report["result_directory"])
    watched.update({
        output / name: value for name, value in report["output_hashes"].items()
    })
    check_unchanged(watched)

    for source in GENERATORS:
        info = report["source_models"][source]
        model_path = Path(info["model_path"]).resolve()
        if (
            not model_path.is_relative_to(ROOT.resolve())
            or watched.get(model_path) != info["model_sha256"]
        ):
            raise ValueError("Saved model provenance does not match the baseline.")

    return report, output / "cross_model_predictions.csv", watched


def prepare_targets(manifest):
    groups = defaultdict(set)
    for row in manifest:
        groups[row["group_id"]].add(row["partition"])
    if any(len(partitions) != 1 for partitions in groups.values()):
        raise ValueError("A similarity group spans partitions.")

    return {
        (partition, generator): {
            row["record_id"]: row for row in manifest
            if row["partition"] == partition
            and (row["origin"] == "human" or row["source"] == generator)
        }
        for partition in ("calibration", "test")
        for generator in GENERATORS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=CLASSIFIERS, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    baseline, predictions_path, watched = load_baseline(args.classifier)
    targets = prepare_targets(read_csv(SPLITS / "manifest.csv"))

    for generator in GENERATORS:
        rows = targets["calibration", generator]
        n_llm = sum(int(row["label"]) for row in rows.values())
        if not 0 < n_llm < len(rows):
            raise ValueError("Each calibration target must contain both classes.")
        print(f"{generator}: calibration human={len(rows)-n_llm}, LLM={n_llm}")

    if args.check_only:
        print("Preflight passed. Baseline files and split groups verified.")
        return

    script_path = Path(__file__).resolve()
    watched[script_path] = sha256(script_path)
    run_id = (
        f"{args.classifier}_recalibration_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    )
    cal_ids = set().union(*(targets["calibration", g] for g in GENERATORS))
    data = {
        row["id"]: row for row in read_csv(ROOT / "data" / "corpus_features.csv")
        if row["id"] in cal_ids
    }

    # Choose all nine thresholds before reading test predictions.
    choices = {}
    for source in GENERATORS:
        model_info = baseline["source_models"][source]
        model = joblib.load(model_info["model_path"])
        for target in GENERATORS:
            cohort = targets["calibration", target]
            ids = sorted(cohort)
            y = np.array([int(cohort[i]["label"]) for i in ids])
            X = np.array([[float(data[i][f]) for f in FEATURES] for i in ids])
            if not np.isfinite(X).all():
                raise ValueError("Calibration features contain nonfinite values.")

            t, f1, n_candidates = choose_threshold(y, llm_scores(model, X))
            choices[source, target] = {
                "threshold": t,
                "calibration_f1": f1,
                "calibration_records": len(ids),
                "calibration_llm_records": int(y.sum()),
                "threshold_candidates": n_candidates,
            }

    old_cells = {
        (row["training_generator"], row["evaluation_generator"]): row
        for row in baseline["cell_metrics"]
    }
    test_rows = defaultdict(list)
    for row in read_csv(predictions_path):
        test_rows[row["training_generator"], row["evaluation_generator"]].append(row)
    if set(test_rows) != set(PAIRS) or set(old_cells) != set(PAIRS):
        raise ValueError("Expected all nine generator pairs.")

    cells, predictions = [], []
    for source, target in PAIRS:
        pair = source, target
        rows = test_rows[pair]
        cohort = targets["test", target]
        ids = [row["record_id"] for row in rows]
        if len(ids) != len(cohort) or set(ids) != set(cohort):
            raise ValueError(f"Test membership differs for {pair}.")

        y = np.array([int(cohort[i]["label"]) for i in ids])
        scores = np.array([float(row["llm_score"]) for row in rows])
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Invalid saved scores for {pair}.")

        before = (scores >= THRESHOLD).astype(int)
        if any(
            int(row["actual_label"]) != actual
            or int(row["predicted_label"]) != pred
            or row["source_experiment_id"] != baseline["source_models"][source]["experiment_id"]
            or row["experiment_id"] != baseline["experiment_id"]
            for row, actual, pred in zip(rows, y, before)
        ):
            raise ValueError(f"Saved predictions differ from baseline for {pair}.")

        old = old_cells[pair]
        if any(
            not np.isclose(value, old[key], rtol=0, atol=1e-12)
            for key, value in label_metrics(y, before).items()
        ):
            raise ValueError(f"Default metrics differ from baseline for {pair}.")

        chosen = choices[pair]
        after = (scores >= chosen["threshold"]).astype(int)
        updated = label_metrics(y, after)
        cells.append({
            "training_generator": source,
            "evaluation_generator": target,
            "test_records": len(rows),
            "default_threshold": THRESHOLD,
            **chosen,
            **updated,
            "roc_auc": old["roc_auc"],
            "average_precision": old["average_precision"],
            "default_f1": old["f1"],
            "f1_change": updated["f1"] - old["f1"],
            "default_fp": old["fp"],
            "default_fn": old["fn"],
            "llm_misses_recovered": int(np.sum((y == 1) & (before == 0) & (after == 1))),
            "llm_newly_missed": int(np.sum((y == 1) & (before == 1) & (after == 0))),
            "human_newly_flagged": int(np.sum((y == 0) & (before == 0) & (after == 1))),
            "human_flags_cleared": int(np.sum((y == 0) & (before == 1) & (after == 0))),
        })

        for row, actual, pred in zip(rows, y, after):
            predictions.append({
                **row,
                "experiment_id": run_id,
                "baseline_experiment_id": baseline["experiment_id"],
                "default_predicted_label": row["predicted_label"],
                "predicted_label": int(pred),
                "threshold": chosen["threshold"],
                "outcome": OUTCOME[int(actual), int(pred)],
                "correct": bool(actual == pred),
            })

    check_unchanged(watched)
    output = ROOT / "results" / "recalibration" / args.classifier / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "recalibration_metrics.csv", list(cells[0]), cells)
    write_csv(output / "recalibration_predictions.csv", list(predictions[0]), predictions)

    default_gap = gap_summary(list(old_cells.values()))
    new_gap = gap_summary(cells)
    report = {
        "experiment_id": run_id,
        "status": "recalibration_complete",
        "evaluation_method": "recalibration",
        "protocol_version": "threshold_recalibration_v1",
        "classifier": args.classifier,
        "baseline_experiment_id": baseline["experiment_id"],
        "source_models": baseline["source_models"],
        "selection_rule": RULE,
        "candidate_rule": "Unique calibration scores plus 0.5. Positive if score >= cutoff.",
        "features": list(FEATURES),
        "split_version": baseline["split_version"],
        "fit_count": 0,
        "versions": baseline["versions"],
        "calibration_records_used": len(cal_ids),
        "final_test_records_used": len(set().union(*(targets["test", g] for g in GENERATORS))),
        "prediction_rows": len(predictions),
        "cell_metrics": cells,
        "default_gap": default_gap,
        "recalibrated_gap": new_gap,
        "gap_reduction_percentage_points": (
            default_gap["performance_gap_percentage_points"]
            - new_gap["performance_gap_percentage_points"]
        ),
        "input_hashes": {str(path): value for path, value in watched.items()},
        "result_directory": str(output),
        "output_hashes": {p.name: sha256(p) for p in sorted(output.iterdir())},
    }

    check_unchanged(watched)
    ledger = ROOT / "ledger" / "recalibration" / args.classifier / f"{run_id}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    for cell in cells:
        print(
            f"{cell['training_generator']} -> {cell['evaluation_generator']} "
            f"threshold={cell['threshold']:.6f}, "
            f"F1 {cell['default_f1']:.4f} -> {cell['f1']:.4f}"
        )
    print("Default gap (pp):", default_gap["performance_gap_percentage_points"])
    print("Recalibrated gap (pp):", new_gap["performance_gap_percentage_points"])
    print("Experiment record:", ledger)


if __name__ == "__main__":
    main()