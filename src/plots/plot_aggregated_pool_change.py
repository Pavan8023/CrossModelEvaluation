"""Plot pooled F1 minus mean transfer F1 from the other two training generators."""

from plot_utils import (
    Inputs, MODELS, GENERATORS, MODEL_NAMES, GENERATOR_NAMES, COLORS,
    arguments, style, heading, clean_axes, finish, plt, np,
)


def main():
    args = arguments(__doc__)
    inputs, plotted = Inputs(__file__), []
    for model in MODELS:
        cross = inputs.run("cross_model", model)
        pool = inputs.run("aggregated_pool", model)
        individual = inputs.rows(cross, "cross_model_metrics.csv")
        pooled = inputs.rows(pool, "aggregated_pool_metrics.csv")
        for target in GENERATORS:
            row = next(r for r in pooled if r["test_scope"] == target)
            others = [r for r in individual if r["evaluation_generator"] == target and r["training_generator"] != target]
            if len(others) != 2 or len({r["training_generator"] for r in others}) != 2:
                raise ValueError("Expected the two different-generator models.")
            for other in others:
                for key in ("human_test_records", "llm_test_records"):
                    if int(row[key]) != int(other[key]):
                        raise ValueError("Pooled and transfer test denominators differ.")
            average = float(np.mean([float(r["f1"]) for r in others]))
            pooled_f1 = float(row["f1"])
            plotted.append({"classifier": model, "test_generator": target,
                            "pooled_generator_f1": pooled_f1,
                            "other_training_generators": ";".join(r["training_generator"] for r in others),
                            "individual_f1_1": float(others[0]["f1"]), "individual_f1_2": float(others[1]["f1"]),
                            "mean_different_generator_f1": average,
                            "change_percentage_points": 100 * (pooled_f1 - average),
                            "human_test_records": int(row["human_test_records"]),
                            "llm_test_records": int(row["llm_test_records"])})
    style()
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 4.7))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.22, top=0.75, wspace=0.20)
    heading(fig, "How much did pooled training change cross-generator F1?",
            "Pooled F1 minus the mean F1 of the two single-generator models trained on other generators")
    for index, target in enumerate(GENERATORS):
        ax = axes[index]
        rows = [next(r for r in plotted if r["classifier"] == model and r["test_generator"] == target) for model in MODELS]
        values = [r["change_percentage_points"] for r in rows]
        ax.barh(range(5), values, height=0.55, color=[COLORS[m] for m in MODELS])
        for y, value in enumerate(values):
            ax.text(value + (0.6 if value >= 0 else -0.6), y, f"{value:+.1f}",
                    ha="left" if value >= 0 else "right", va="center", color=COLORS[MODELS[y]], fontsize=9, weight="bold")
        ax.axvline(0, color="#7c8794", linewidth=0.8)
        ax.set(xlim=(-6, 55), ylim=(4.5, -0.5))
        ax.set_xticks(range(0, 51, 10))
        ax.set_yticks(range(5), [MODEL_NAMES[m] for m in MODELS] if index == 0 else [])
        ax.set_title(GENERATOR_NAMES[target], pad=9)
        clean_axes(ax)
    fig.text(0.57, 0.08, "F1 change (percentage points)", ha="center", fontsize=10)
    finish(fig, "aggregated_pool/pooled_training_f1_change", plotted, inputs,
           "For each target generator: 100 * (pooled model generator-specific F1 minus equal-weight mean F1 of the two individual models trained on other generators). This is not pooled-vs-matched training. Pooled training includes the target generator and more records; this is not a causal estimate of diversity alone.", args.check_only)


if __name__ == "__main__":
    main()
