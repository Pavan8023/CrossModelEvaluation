"""Run the established evaluation protocols for the additional benchmarks."""

import argparse
import json
import math
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from evaluate_cross_model import GENERATORS
from feature_validation import sha256
from train_knn import ROOT

CLASSIFIERS = ("xgboost", "random_forest")
METHODS = (
    "intra_model",
    "cross_model",
    "recalibration",
    "cross_dataset",
    "aggregated_pool",
)


def package_version(name):
    return version("scikit-learn" if name == "random_forest" else name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--classifier", choices=CLASSIFIERS, required=True)
    parser.add_argument("--generator", choices=GENERATORS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if (args.method == "intra_model") != (args.generator is not None):
        parser.error("--generator is required for intra_model and applies only there")

    ledger_dir = ROOT / "ledger" / args.method / args.classifier
    pattern = f"*_{args.generator}_*.json" if args.generator else "*.json"
    if not args.check_only and any(ledger_dir.glob(pattern)):
        parser.error(f"A ledger already exists for this evaluation in {ledger_dir}")

    evaluation = import_module(f"evaluate_{args.method}")
    launcher_path = Path(__file__).resolve()
    launcher_hash = sha256(launcher_path)
    check_inputs = evaluation.check_unchanged

    def check_with_launcher(watched):
        if watched.setdefault(launcher_path, launcher_hash) != launcher_hash:
            raise ValueError("Benchmark launcher differs from the recorded version")
        check_inputs(watched)

    def dump_report(report, stream, **kwargs):
        report["versions"].setdefault(
            args.classifier, package_version(args.classifier)
        )
        for fold in report.get("fold_configurations", []):
            params = fold["fitted_estimator_parameters"]
            for key, value in params.items():
                if isinstance(value, float) and not math.isfinite(value):
                    params[key] = str(value)
        json.dump(report, stream, allow_nan=False, **kwargs)

    # Configure this evaluator for the current command.
    evaluation.CLASSIFIERS = CLASSIFIERS
    evaluation.check_unchanged = check_with_launcher
    evaluation.json = SimpleNamespace(
        loads=json.loads, dumps=json.dumps, dump=dump_report
    )
    if hasattr(evaluation, "metadata"):
        evaluation.metadata = SimpleNamespace(version=package_version)
    if hasattr(evaluation, "version"):
        evaluation.version = package_version

    sys.argv = [sys.argv[0], "--classifier", args.classifier]
    if args.generator:
        sys.argv.extend(("--generator", args.generator))
    if args.check_only:
        sys.argv.append("--check-only")
    evaluation.main()


if __name__ == "__main__":
    main()