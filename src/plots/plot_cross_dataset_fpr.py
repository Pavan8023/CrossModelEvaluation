"""Plot human-source false-positive rates for source-included/held-out models."""

from matplotlib.patches import Patch
from plot_utils import (
    Inputs, MODELS, MODEL_NAMES, arguments, close, style, heading,
    clean_axes, finish, plt, np,
)

SOURCES = ("CEAS_08", "TREC-07", "Nazario", "Nigerian_Fraud", "enron_data_fraud_labeled")
SOURCE_NAMES = ("CEAS-08", "TREC-07", "Nazario", "Nigerian Fraud", "Enron subset")


def main():
    args = arguments(__doc__)
    inputs, plotted = Inputs(__file__), []
    for model in MODELS:
        run = inputs.run("cross_dataset", model)
        comparisons = inputs.rows(run, "cross_dataset_source_comparison.csv")
        cells = inputs.rows(run, "cross_dataset_metrics.csv")
        for source in SOURCES:
            row = next(r for r in comparisons if r["human_source"] == source)
            source_cells = [r for r in cells if r["test_human_source"] == source]
            included = [r for r in source_cells if r["excluded_human_source"] != source]
            excluded = [r for r in source_cells if r["excluded_human_source"] == source]
            if len(included) != 4 or len(excluded) != 1:
                raise ValueError("Expected four source-included models and one source-holdout model.")
            for cell in source_cells:
                close(cell["human_false_positive_rate"], int(cell["fp"]) / int(cell["test_human_records"]), "Source FPR")
            seen = float(np.mean([float(r["human_false_positive_rate"]) for r in included]))
            unseen = float(excluded[0]["human_false_positive_rate"])
            close(seen, row["mean_seen_fpr"], "Mean included-source FPR")
            close(unseen, row["unseen_fpr"], "Held-out source FPR")
            plotted.append({"classifier": model, "human_source": source,
                            "human_test_records": int(excluded[0]["test_human_records"]),
                            "included_mean_fpr_percent": 100 * seen, "held_out_fpr_percent": 100 * unseen,
                            "difference_percentage_points": 100 * (unseen - seen)})
    style()
    fig, axes = plt.subplots(2, 3, figsize=(9.1, 6.4))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.14, top=0.78, wspace=0.22, hspace=0.62)
    heading(fig, "Nazario had the largest rise in false alarms when held out",
            "Human-written phishing emails flagged as generated / Source included is the mean of four models")
    for index, model in enumerate(MODELS):
        ax = axes.flat[index]
        rows = [next(r for r in plotted if r["classifier"] == model and r["human_source"] == source) for source in SOURCES]
        y = np.arange(5)
        ax.barh(y - 0.14, [r["included_mean_fpr_percent"] for r in rows], height=0.25, color="#126782")
        ax.barh(y + 0.14, [r["held_out_fpr_percent"] for r in rows], height=0.25, color="#b54720")
        nazario = rows[2]
        for key, offset, color in (("included_mean_fpr_percent", -0.14, "#126782"), ("held_out_fpr_percent", 0.14, "#b54720")):
            value = nazario[key]
            label = f"{value:.3f}".rstrip("0").rstrip(".") + "%"
            ax.text(value + 0.7, 2 + offset, label, ha="left", va="center", fontsize=8, color=color, weight="bold")
        ax.set(xlim=(0, 60), ylim=(4.6, -0.6))
        ax.set_xticks([0, 15, 30, 45, 60])
        ax.set_yticks(y, SOURCE_NAMES if index % 3 == 0 else [])
        ax.set_title(MODEL_NAMES[model], pad=9)
        clean_axes(ax)
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles=[Patch(color="#126782", label="Source included (mean)"),
                             Patch(color="#b54720", label="Source held out")], loc="center", frameon=False, fontsize=9)
    fig.text(0.57, 0.045, "Human-email false-positive rate (%)", ha="center", fontsize=10)
    finish(fig, "cross_dataset/human_source_false_positive_rates", plotted, inputs,
           "For each source, compare the held-out model FPR against the equal-weight mean FPR of the four models including that source. All rates use the same held-out emails from that source; training size and source mixture differ.", args.check_only)


if __name__ == "__main__":
    main()
