"""Plot paired default/recalibrated 3-by-3 F1 matrices for five classifiers."""

from matplotlib.colors import Normalize
from plot_utils import (
    Inputs, MODELS, GENERATORS, MODEL_NAMES, GENERATOR_NAMES, TEXT,
    arguments, close, style, heading, finish, plt, np,
)


def main():
    args = arguments(__doc__)
    inputs, plotted, matrices = Inputs(__file__), [], {}
    for model in MODELS:
        base = inputs.run("cross_model", model)
        recal = inputs.run("recalibration", model)
        if recal["baseline_experiment_id"] != base["experiment_id"]:
            raise ValueError("Recalibration does not refer to this baseline.")
        old = inputs.rows(base, "cross_model_metrics.csv")
        new = inputs.rows(recal, "recalibration_metrics.csv")
        for condition, rows in (("default", old), ("recalibrated", new)):
            by_pair = {(r["training_generator"], r["evaluation_generator"]): r for r in rows}
            if len(rows) != 9 or len(by_pair) != 9:
                raise ValueError("Expected nine unique generator pairs.")
            matrix = np.empty((3, 3))
            for i, train in enumerate(GENERATORS):
                for j, test in enumerate(GENERATORS):
                    row = by_pair[train, test]
                    f1 = float(row["f1"])
                    close(f1, 2 * int(row["tp"]) / (2 * int(row["tp"]) + int(row["fp"]) + int(row["fn"])), "Cell F1")
                    if condition == "recalibrated":
                        original = next(r for r in old if r["training_generator"] == train and r["evaluation_generator"] == test)
                        close(row["default_f1"], original["f1"], "Paired default F1")
                        if row["test_records"] != original["test_records"]:
                            raise ValueError("Test denominators differ.")
                    matrix[i, j] = f1
                    plotted.append({"classifier": model, "condition": condition,
                                    "training_generator": train, "test_generator": test,
                                    "threshold": float(row["threshold"]), "f1": f1,
                                    "test_records": int(row["test_records"])})
            matrices[model, condition] = matrix
    style()
    fig, axes = plt.subplots(5, 2, figsize=(9.1, 8.2))
    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.13, top=0.80, wspace=0.15, hspace=0.20)
    heading(fig, "Cross-generator F1 before and after recalibration",
            "The test emails and scores are held fixed / Bold cells use the same generator for training and testing")
    cmap, norm = plt.get_cmap("cividis"), Normalize(vmin=0.4, vmax=1.0)
    for i, model in enumerate(MODELS):
        for j, condition in enumerate(("default", "recalibrated")):
            ax, matrix = axes[i, j], matrices[model, condition]
            mesh = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(3), [GENERATOR_NAMES[g] for g in GENERATORS])
            ax.set_yticks(range(3), [GENERATOR_NAMES[g] for g in GENERATORS] if j == 0 else [])
            ax.tick_params(length=0, labelsize=8, labelbottom=i == 4)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for y in range(3):
                for x in range(3):
                    ax.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center",
                            fontsize=9, color="white" if matrix[y, x] < 0.74 else TEXT,
                            weight="bold" if x == y else "normal")
            if i == 0:
                ax.set_title("Default threshold (0.5)" if j == 0 else "Target-recalibrated threshold", pad=10)
        box = axes[i, 0].get_position()
        fig.text(0.075, (box.y0 + box.y1) / 2, MODEL_NAMES[model], rotation=90,
                 ha="center", va="center", fontsize=9, weight="bold")
    cax = fig.add_axes([0.91, 0.13, 0.018, 0.67])
    fig.colorbar(mesh, cax=cax, ticks=np.linspace(0.4, 1.0, 5)).set_label("F1")
    fig.text(0.02, 0.47, "Training generator", rotation=90, ha="center", va="center", fontsize=10)
    fig.text(0.53, 0.045, "Test generator", ha="center", fontsize=10)
    finish(fig, "cross_model/cross_model_f1_before_after_recalibration", plotted, inputs,
           "Per-cell held-out F1; shared 0.4-to-1.0 color scale. Default and recalibrated panels use the same test emails and scores. Recalibrated thresholds were selected previously on separate calibration data; this script selects no thresholds.", args.check_only)


if __name__ == "__main__":
    main()
