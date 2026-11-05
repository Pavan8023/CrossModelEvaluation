"""Shared style, verified inputs, and provenance for the six paper figures.

These plotting scripts reconstruct the earlier one-off figures from saved
experiment outputs. They never train, tune, or load an authorship detector.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save files without opening a graphical window.
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PLAN = Path(__file__).with_name("figure_sources.json")
MODELS = ("knn", "lightgbm", "catboost", "xgboost", "random_forest")
GENERATORS = ("gpt-4.1", "deepseek3.2", "llama-3.3-70b")
MODEL_NAMES = dict(zip(MODELS, ("KNN", "LightGBM", "CatBoost", "XGBoost", "Random Forest")))
GENERATOR_NAMES = dict(zip(GENERATORS, ("GPT-4.1", "DeepSeek", "Llama")))
COLORS = dict(zip(MODELS, ("#126782", "#b54720", "#3d6b4b", "#6b5091", "#7a5415")))
TEXT, MUTED, GRID = "#172337", "#50637d", "#e2e8f0"
DPI = 450


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label, tolerance=1e-11):
    """Raise an error rather than silently plotting inconsistent numbers."""
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label}: {actual} != {expected}")


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--check-only", action="store_true", help="Verify inputs and calculations without writing files.")
    return parser.parse_args()


class Inputs:
    """Use explicitly pinned ledgers and verify every plotted CSV's SHA-256."""

    def __init__(self, script):
        self.script = Path(script).resolve()
        self.files = {}
        self.experiments = {}
        self.track(self.script)
        self.track(Path(__file__))
        self.track(SOURCE_PLAN)
        self.plan = json.loads(SOURCE_PLAN.read_text(encoding="utf-8"))

    def track(self, path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Input is outside the research project: {path}")
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Input hash differs from its saved record: {path}")
        previous = self.files.setdefault(path, actual)
        if previous != actual:
            raise ValueError(f"Input changed during plotting: {path}")
        return path

    def run(self, stage, model, generator=None):
        key = "|".join((stage, model, generator or "all"))
        spec = self.plan["runs"][key]
        path = self.track(ROOT / spec["path"], spec["sha256"])
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["classifier"] != model:
            raise ValueError(f"Unexpected classifier in {path}")
        self.experiments[key] = {"experiment_id": report["experiment_id"], "ledger": path.relative_to(ROOT).as_posix()}
        return report

    def rows(self, report, filename):
        # Relative paths keep plotting portable when the project is relocated.
        stage = report.get("evaluation_method", "shap")
        path = ROOT / "results" / stage / report["classifier"] / report["experiment_id"] / filename
        self.track(path, report["output_hashes"][filename])
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def verify(self):
        for path, expected in self.files.items():
            if sha256(path) != expected:
                raise ValueError(f"Input changed during plotting: {path}")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "cme-paper-figures-v1", "figure.dpi": 150,
    })


def heading(fig, title, subtitle):
    fig.text(0.045, 0.96, title, ha="left", va="top", fontsize=17, weight="bold")
    fig.text(0.045, 0.89, subtitle, ha="left", va="top", fontsize=10, color=MUTED)


def clean_axes(ax, grid="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=3)


def finish(fig, relative_stem, data, inputs, notes, check_only=False):
    """Save the figure, its full-precision plotted data, and a source record."""
    inputs.verify()
    if check_only:
        plt.close(fig)
        print(f"Verified {relative_stem}: {len(data)} plotted rows; no files written.")
        return
    stem = ROOT / "figures" / relative_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(png, dpi=DPI, metadata={"Software": f"Matplotlib {matplotlib.__version__}"})
    fig.savefig(svg, metadata={"Date": None, "Creator": f"Matplotlib {matplotlib.__version__}"})
    values_path = stem.with_suffix(".data.csv")
    with values_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    report = {
        "purpose": "Reconstructed, reproducible plotting of existing experiment results",
        "plotting_script": inputs.script.relative_to(ROOT).as_posix(),
        "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "dpi": DPI, "figure_size_inches": fig.get_size_inches().tolist(),
        "experiments": inputs.experiments,
        "source_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": h} for p, h in sorted(inputs.files.items())],
        "outputs": {p.relative_to(ROOT).as_posix(): sha256(p) for p in (png, svg, values_path)},
        "calculation": notes, "model_fits": 0,
    }
    stem.with_suffix(".sources.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    inputs.verify()
    plt.close(fig)
    print(f"Saved {png.relative_to(ROOT)} and SVG, data CSV, source JSON.")
