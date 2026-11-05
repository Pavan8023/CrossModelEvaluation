"""Plot paired FN/FP counts before and after DeepSeek-to-Llama recalibration."""

from matplotlib.lines import Line2D
from plot_utils import (
    Inputs, MODELS, MODEL_NAMES, TEXT, arguments, close, style,
    heading, clean_axes, finish, plt, np,
)


def main():
    args = arguments(__doc__)
    inputs, plotted = Inputs(__file__), []
    for model in MODELS:
        base = inputs.run("cross_model", model)
        recal = inputs.run("recalibration", model)
        if recal["baseline_experiment_id"] != base["experiment_id"]:
            raise ValueError("Unpaired recalibration run.")
        def pair(rows):
            return next(r for r in rows if r["training_generator"] == "deepseek3.2" and r["evaluation_generator"] == "llama-3.3-70b")
        old = pair(inputs.rows(base, "cross_model_metrics.csv"))
        new = pair(inputs.rows(recal, "recalibration_metrics.csv"))
        if old["test_records"] != new["test_records"]:
            raise ValueError("The test sets have different denominators.")
        close(new["default_f1"], old["f1"], "Baseline F1")
        for metric in ("fp", "fn"):
            if int(new["default_" + metric]) != int(old[metric]):
                raise ValueError("The stored default confusion counts differ.")
        plotted.append({"classifier": model, "training_generator": "deepseek3.2",
                        "test_generator": "llama-3.3-70b", "human_records": int(old["human_test_records"]),
                        "llm_records": int(old["llm_test_records"]), "default_fn": int(old["fn"]),
                        "recalibrated_fn": int(new["fn"]), "default_fp": int(old["fp"]),
                        "recalibrated_fp": int(new["fp"]), "threshold": float(new["threshold"]),
                        "default_f1": float(old["f1"]), "recalibrated_f1": float(new["f1"])})
    if len({(r["human_records"], r["llm_records"]) for r in plotted}) != 1:
        raise ValueError("Classifier test counts differ.")
    humans, llms = plotted[0]["human_records"], plotted[0]["llm_records"]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 5.0))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.23, top=0.76, wspace=0.39)
    heading(fig, "Recalibration recovered Llama emails and raised false alarms",
            f"DeepSeek to Llama / Same {llms:,} Llama emails and {humans:,} human emails per comparison")
    for ax, metric, title, color in zip(axes, ("fn", "fp"),
            ("Missed Llama emails", "Human emails flagged"), ("#126782", "#b54720")):
        for y, row in enumerate(plotted):
            before, after = row["default_" + metric], row["recalibrated_" + metric]
            ax.annotate("", xy=(after, y), xytext=(before, y),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8, "mutation_scale": 11})
            ax.scatter(before, y, s=42, facecolors="white", edgecolors=TEXT, zorder=3)
            ax.scatter(after, y, marker="D", s=32, color=color, zorder=4)
            ax.annotate(str(before), (before, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, weight="bold")
            ax.annotate(str(after), (after, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, weight="bold")
        maximum = max(r["default_" + metric] for r in plotted) if metric == "fn" else max(r["recalibrated_" + metric] for r in plotted)
        limit = 50 * np.ceil((maximum + 5) / 50) if metric == "fn" else 20 * np.ceil((maximum + 5) / 20)
        ax.set(xlim=(-3, limit + 2), ylim=(4.5, -0.5), xlabel="Emails")
        ax.set_yticks(range(5), [MODEL_NAMES[m] for m in MODELS])
        ax.set_title(title, pad=12)
        clean_axes(ax)
    handles = [Line2D([], [], marker="o", color="none", markeredgecolor=TEXT, markerfacecolor="white", label="Default"),
               Line2D([], [], marker="D", color="none", markerfacecolor=TEXT, markeredgecolor=TEXT, label="Recalibrated")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False)
    finish(fig, "recalibration/deepseek_to_llama_threshold_tradeoff", plotted, inputs,
           "Paired confusion counts at the default and previously selected calibration thresholds. FP is a human-written phishing email called LLM-generated; FN is an LLM-generated phishing email called human-written.", args.check_only)


if __name__ == "__main__":
    main()
