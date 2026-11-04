"""Summarize errors on paired held-out predictions."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "splits_v2" / "manifest.csv"
CLASSIFIERS = ("knn", "lightgbm", "catboost")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == expected, f"SHA-256 mismatch: {path}")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_ledger(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prediction_index(ledger, method, classifier):
    path = Path(ledger["result_directory"]) / f"{method}_predictions.csv"
    verify_hash(path, ledger["output_hashes"][path.name])
    indexed = {}
    for row in read_rows(path):
        key = (row["training_generator"], row["evaluation_generator"], row["record_id"])
        require(key not in indexed, f"Duplicate prediction: {key}")
        require(row["classifier"] == classifier
                and row["experiment_id"] == ledger["experiment_id"],
                f"Unexpected run or classifier: {key}")
        indexed[key] = row
    return indexed


def summarize(classifier):
    recal_dir = ROOT / "ledger" / "recalibration" / classifier
    recal_path = max(recal_dir.glob("*.json"))
    recal = read_ledger(recal_path)
    baseline_id = recal["baseline_experiment_id"]
    base_path = ROOT / "ledger" / "cross_model" / classifier / f"{baseline_id}.json"
    base = read_ledger(base_path)
    verify_hash(base_path, recal["input_hashes"][str(base_path)])
    manifest_hash = recal["input_hashes"][str(MANIFEST)]
    verify_hash(MANIFEST, manifest_hash)
    require(manifest_hash == base["input_hashes"][str(MANIFEST)],
            "The runs used different split manifests")

    old = prediction_index(base, "cross_model", classifier)
    new = prediction_index(recal, "recalibration", classifier)
    test_rows = [row for row in read_rows(MANIFEST) if row["partition"] == "test"]
    test = {row["record_id"]: row for row in test_rows}
    require(len(test) == len(test_rows), "Duplicate test ID in the split manifest")
    generators = sorted({row["source"] for row in test_rows if row["label"] == "1"})
    expected = {
        (train, target, row["record_id"])
        for train, target in product(generators, repeat=2)
        for row in test_rows if row["label"] == "0" or row["source"] == target
    }
    require(old.keys() == new.keys() == expected, "Unexpected test coverage")
    cells = {
        (row["training_generator"], row["evaluation_generator"]): row
        for row in recal["cell_metrics"]
    }

    by_group = defaultdict(Counter)
    by_pair = defaultdict(Counter)
    for key, before in old.items():
        train, target, record_id = key
        after, meta = new[key], test[record_id]
        require(all(before[field] == after[field] for field in
                    ("llm_score", "actual_label", "email_source", "group_id"))
                and before["group_id"] == meta["group_id"]
                and before["email_source"] == meta["source"]
                and before["actual_label"] == meta["label"]
                and before["predicted_label"] == after["default_predicted_label"]
                and after["baseline_experiment_id"] == baseline_id,
                f"Prediction alignment failed: {key}")

        label = int(meta["label"])
        old_wrong = int(int(before["predicted_label"]) != label)
        new_wrong = int(int(after["predicted_label"]) != label)
        group_type, group = ("llm_theme", meta["category"]) if label else (
            "human_corpus", meta["source"]
        )
        require(bool(group.strip()), f"Missing subgroup: {key}")
        by_group[train, target, group_type, group][old_wrong, new_wrong] += 1
        by_pair[train, target, label][old_wrong, new_wrong] += 1

    for train, target in product(generators, repeat=2):
        llm = by_pair[train, target, 1]
        human = by_pair[train, target, 0]
        actual = {
            "default_fn": llm[1, 0] + llm[1, 1],
            "fn": llm[0, 1] + llm[1, 1],
            "llm_misses_recovered": llm[1, 0],
            "llm_newly_missed": llm[0, 1],
            "default_fp": human[1, 0] + human[1, 1],
            "fp": human[0, 1] + human[1, 1],
            "human_flags_cleared": human[1, 0],
            "human_newly_flagged": human[0, 1],
        }
        cell = cells[train, target]
        require(all(value == cell[name] for name, value in actual.items()),
                f"Totals disagree with ledger: {train} -> {target}")

    summary = []
    for (train, target, kind, group), counts in sorted(by_group.items()):
        total = sum(counts.values())
        before_errors = counts[1, 0] + counts[1, 1]
        after_errors = counts[0, 1] + counts[1, 1]
        summary.append({
            "classifier": classifier,
            "training_generator": train,
            "evaluation_generator": target,
            "group_type": kind,
            "group": group,
            "tested": total,
            "default_errors": before_errors,
            "recalibrated_errors": after_errors,
            "default_error_pct": round(100 * before_errors / total, 4),
            "recalibrated_error_pct": round(100 * after_errors / total, 4),
            "errors_cleared": counts[1, 0],
            "new_errors": counts[0, 1],
            "errors_at_both_thresholds": counts[1, 1],
            "baseline_experiment_id": baseline_id,
            "recalibration_experiment_id": recal["experiment_id"],
            "manifest_sha256": manifest_hash,
        })
    return len(old), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", choices=CLASSIFIERS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    classifiers = (args.classifier,) if args.classifier else CLASSIFIERS
    checked = {classifier: summarize(classifier) for classifier in classifiers}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    for classifier, (paired, rows) in checked.items():
        print(f"{classifier}: {paired:,} paired predictions, {len(rows)} subgroup rows")
        if args.check_only:
            continue
        output = ROOT / "results" / "error_analysis" / classifier / run_id
        output.mkdir(parents=True, exist_ok=False)
        path = output / "threshold_errors_by_subgroup.csv"
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()