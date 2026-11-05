"""Plot five-fold nested intra-model F1 means and sample standard deviations."""

from plot_utils import (
    Inputs, MODELS, GENERATORS, MODEL_NAMES, GENERATOR_NAMES, COLORS,
    arguments, close, style, heading, clean_axes, finish, plt, np,
)


def main():
    args = arguments(__doc__)
    inputs, plotted = Inputs(__file__), []
    for model in MODELS:
        for generator in GENERATORS:
            run = inputs.run("intra_model", model, generator)
            folds = inputs.rows(run, "intra_model_fold_metrics.csv")
            if sorted(int(row["outer_fold"]) for row in folds) != list(range(5)):
                raise ValueError("Expected exactly five distinct outer folds.")
            scores = np.array([float(row["f1"]) for row in folds])
            average, spread = float(scores.mean()), float(scores.std(ddof=1))
            close(average, run["outer_fold_metrics"]["f1"]["mean"], "Outer F1 mean")
            close(spread, run["outer_fold_metrics"]["f1"]["std"], "Outer F1 sample SD")
            plotted.append({"classifier": model, "generator": generator,
                            "mean_f1": average, "sample_sd_f1": spread, "outer_folds": len(folds)})
    style()
    fig, ax = plt.subplots(figsize=(9.1, 4.85))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.26, top=0.76)
    heading(fig, "Within-generator performance across five classifiers",
            "Nested cross-validation / Points show mean F1; whiskers show one standard deviation across outer folds")
    for model, offset, marker in zip(MODELS, np.linspace(-0.20, 0.20, 5), ("o", "s", "D", "^", "P")):
        rows = [next(r for r in plotted if r["classifier"] == model and r["generator"] == g) for g in GENERATORS]
        ax.errorbar(np.arange(3) + offset, [r["mean_f1"] for r in rows],
                    yerr=[r["sample_sd_f1"] for r in rows], fmt=marker,
                    color=COLORS[model], markersize=5, capsize=3, linewidth=1.2,
                    label=MODEL_NAMES[model])
    ax.set_xticks(range(3), [GENERATOR_NAMES[g] for g in GENERATORS])
    ax.set(ylim=(0.92, 1.003), ylabel="F1", xlabel="Email generator")
    ax.set_yticks(np.arange(0.92, 1.001, 0.02))
    clean_axes(ax, "y")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=5, frameon=False, fontsize=9)
    finish(fig, "intra_model/summary/intra_model_f1_five_classifiers", plotted, inputs,
           "Mean and sample SD (ddof=1) of five outer-fold F1 scores. SD is not a confidence interval. Equal generator weighting is not applied within individual plotted points.", args.check_only)


if __name__ == "__main__":
    main()
