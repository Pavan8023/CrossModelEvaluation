"""Explain existing CatBoost scores without fitting any models."""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from catboost import Pool

from evaluate_cross_model import GENERATORS
from feature_validation import FEATURES, read_csv, sha256
from train_knn import ROOT, THRESHOLD, write_csv
from train_lightgbm import check_unchanged

TRAIN_SOURCE = "deepseek3.2"
TEST_SOURCE = "llama-3.3-70b"


def verify(path, expected, watched):
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Expected a file inside the project: {path}")
    if sha256(path) != expected:
        raise ValueError(f"File hash mismatch: {path}")
    watched[path] = expected
    return path


def latest_ledger(method, pattern, watched):
    folder = ROOT / "ledger" / method / "catboost"
    path = max(folder.glob(pattern))
    verify(path, sha256(path), watched)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["classifier"] != "catboost" or report["features"] != list(FEATURES):
        raise ValueError(f"Unexpected classifier or feature order: {path}")
    return report


def artifact(report, name, watched):
    path = Path(report["result_directory"]) / name
    return verify(path, report["output_hashes"][name], watched)


def explain(rows, model_path, run_id, analysis, group):
    pipe = joblib.load(model_path)
    if list(pipe.named_steps) != ["catboost"] or list(pipe.classes_) != [0, 1]:
        raise ValueError(f"Unexpected CatBoost pipeline: {model_path}")
    model = pipe.named_steps["catboost"]
    X = np.array([[float(row[f]) for f in FEATURES] for row in rows])
    if not np.isfinite(X).all():
        raise ValueError("Non-finite feature value")

    values = np.asarray(model.get_feature_importance(
        data=Pool(X), type="ShapValues", thread_count=1
    ))
    raw = model.predict(X, prediction_type="RawFormulaVal", thread_count=1)
    score = pipe.predict_proba(X)[:, 1]
    if values.shape != (len(rows), len(FEATURES) + 1):
        raise ValueError("Unexpected SHAP matrix shape")
    np.testing.assert_allclose(values.sum(axis=1), raw, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        score, [float(r["llm_score"]) for r in rows], rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        score >= THRESHOLD, [int(r["predicted_label"]) for r in rows]
    )

    return [{
        "analysis": analysis,
        "group": group,
        "source_experiment_id": run_id,
        "model_file": model_path.name,
        "record_id": row["record_id"],
        "group_id": row["group_id"],
        "actual_label": int(row["actual_label"]),
        "predicted_label": int(row["predicted_label"]),
        "llm_score": float(score[i]),
        "base_raw_score": float(values[i, -1]),
        "raw_score": float(raw[i]),
        **{f"shap_{f}": float(values[i, j]) for j, f in enumerate(FEATURES)},
    } for i, row in enumerate(rows)]


def summarize(rows):
    matrix = np.array([[r[f"shap_{f}"] for f in FEATURES] for r in rows])
    magnitudes = np.abs(matrix).mean(axis=0)
    order = sorted(range(len(FEATURES)), key=lambda i: (-magnitudes[i], FEATURES[i]))
    return [{
        "analysis": rows[0]["analysis"],
        "group": rows[0]["group"],
        "records": len(rows),
        "rank": rank,
        "feature": FEATURES[i],
        "mean_abs_shap": float(magnitudes[i]),
        "mean_signed_shap": float(matrix[:, i].mean()),
        "negative_share": float(np.mean(matrix[:, i] < 0)),
    } for rank, i in enumerate(order, 1)]


def collect(watched):
    for generator in GENERATORS:
        report = latest_ledger("intra_model", f"*_{generator}_*.json", watched)
        rows = read_csv(artifact(report, "intra_model_predictions.csv", watched))
        fold_path = Path(report["outer_cv_manifest"])
        verify(fold_path, report["input_hashes"][str(fold_path)], watched)
        expected = {
            r["record_id"]: (r["group_id"], r["validation_fold"])
            for r in read_csv(fold_path)
        }
        assigned = {r["record_id"]: (r["group_id"], r["outer_fold"]) for r in rows}
        if len(rows) != len(assigned) or assigned != expected:
            raise ValueError(f"Outer-fold assignments differ: {generator}")

        explained = []
        for fold in range(report["outer_folds"]):
            subset = [r for r in rows if int(r["outer_fold"]) == fold]
            model_name = f"outer_fold_{fold}_model.joblib"
            if not subset or any(r["model_file"] != model_name for r in subset):
                raise ValueError(f"Unexpected fold model: {generator}, fold {fold}")
            model_path = artifact(report, model_name, watched)
            explained.extend(explain(
                subset, model_path, report["experiment_id"], "intra_oof", generator
            ))
        yield explained

    cross = latest_ledger("cross_model", "*.json", watched)
    rows = read_csv(artifact(cross, "cross_model_predictions.csv", watched))
    source = cross["source_models"][TRAIN_SOURCE]
    model_path = verify(Path(source["model_path"]), source["model_sha256"], watched)
    positives = [r for r in rows if r["training_generator"] == TRAIN_SOURCE
                 and r["evaluation_generator"] == TEST_SOURCE
                 and r["actual_label"] == "1"]
    if (len({r["record_id"] for r in positives}) != len(positives)
            or any(r["source_experiment_id"] != source["experiment_id"]
                   or r["email_source"] != TEST_SOURCE for r in positives)):
        raise ValueError("Unexpected test rows for the cross-generator diagnostic")

    cell = next(r for r in cross["cell_metrics"]
                if r["training_generator"] == TRAIN_SOURCE
                and r["evaluation_generator"] == TEST_SOURCE)
    for prediction, label, metric in (("0", "missed_llama", "fn"),
                                      ("1", "detected_llama", "tp")):
        subset = [r for r in positives if r["predicted_label"] == prediction]
        if not subset or len(subset) != cell[metric]:
            raise ValueError(f"Cross-generator counts disagree: {label}")
        yield explain(subset, model_path, cross["experiment_id"], "deepseek_to_llama", label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    watched, predictions, summaries = {}, [], []
    for rows in collect(watched):
        summary = summarize(rows)
        predictions.extend(rows)
        summaries.extend(summary)
        print(rows[0]["group"], f"({len(rows):,} emails)",
              "top five:", ", ".join(row["feature"] for row in summary[:5]))

    check_unchanged(watched)
    print("SHAP sums and saved scores verified for every explained email.")
    if args.check_only:
        return

    run_id = f"catboost_shap_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    output = ROOT / "results" / "shap" / "catboost" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("shap_by_email.csv", predictions),
                       ("shap_feature_summary.csv", summaries)):
        write_csv(output / name, list(rows[0]), rows)

    report = {
        "experiment_id": run_id,
        "classifier": "catboost",
        "fit_count": 0,
        "method": "CatBoost native ShapValues",
        "score_scale": "raw score (log-odds)",
        "features": list(FEATURES),
        "versions": {name: version(name) for name in ("catboost", "numpy", "joblib")},
        "result_directory": str(output),
        "script_sha256": sha256(Path(__file__)),
        "input_hashes": {str(path): value for path, value in watched.items()},
        "output_hashes": {p.name: sha256(p) for p in output.iterdir()},
    }
    ledger = ROOT / "ledger" / "shap" / "catboost" / f"{run_id}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("Experiment record:", ledger)


if __name__ == "__main__":
    main()