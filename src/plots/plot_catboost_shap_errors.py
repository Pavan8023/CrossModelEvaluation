"""Plot signed CatBoost SHAP contributions on missed versus detected Llama emails.

Uses existing SHAP CSVs; it does not recompute SHAP or fit a model.
"""

from matplotlib.lines import Line2D
from plot_utils import (
    Inputs, MUTED, arguments, close, style, heading, clean_axes,
    finish, plt, np,
)

FEATURE_NAMES = {
    "ttr": "Type-token ratio", "mean_word_len": "Mean word length",
    "mean_sentence_len_tokens": "Mean sentence length", "yules_k": "Yule's K (word repetition)",
    "clause_density": "Clause density", "noun_ratio": "Noun ratio", "verb_ratio": "Verb ratio",
    "mean_parse_depth": "Mean parse depth", "imperative_count": "Imperative count",
    "first_person_ratio": "First-person ratio", "second_person_ratio": "Second-person ratio",
    "politeness_density": "Politeness density", "urgency_density": "Urgency density",
    "url_density": "URL density", "cta_density": "Call-to-action density",
    "authority_density": "Authority density", "time_pressure_density": "Time-pressure density",
}


def main():
    args = arguments(__doc__)
    inputs = Inputs(__file__)
    run = inputs.run("shap", "catboost")
    summary = inputs.rows(run, "shap_feature_summary.csv")
    explanations = inputs.rows(run, "shap_by_email.csv")
    cross = inputs.run("cross_model", "catboost")
    predictions = inputs.rows(cross, "cross_model_predictions.csv")
    target = [r for r in predictions if r["training_generator"] == "deepseek3.2"
              and r["evaluation_generator"] == "llama-3.3-70b" and r["actual_label"] == "1"]
    target_by_id = {r["record_id"]: r for r in target}
    if len(target_by_id) != len(target):
        raise ValueError("Duplicate target prediction IDs.")
    features, groups, means = run["features"], {}, {}
    for cohort, prediction in (("missed_llama", "0"), ("detected_llama", "1")):
        rows = [r for r in explanations if r["analysis"] == "deepseek_to_llama" and r["group"] == cohort]
        expected_ids = {r["record_id"] for r in target if r["predicted_label"] == prediction}
        if not rows or {r["record_id"] for r in rows} != expected_ids or len(rows) != len(expected_ids):
            raise ValueError("SHAP cohort membership differs from the saved predictions.")
        matrix = np.array([[float(r["shap_" + f]) for f in features] for r in rows])
        for row, contributions in zip(rows, matrix):
            if row["source_experiment_id"] != cross["experiment_id"]:
                raise ValueError("The SHAP explanations refer to another cross-model run.")
            close(contributions.sum() + float(row["base_raw_score"]), row["raw_score"], "SHAP additivity", 1e-9)
            close(row["llm_score"], target_by_id[row["record_id"]]["llm_score"], "Saved explanation score")
            close(1 / (1 + np.exp(-float(row["raw_score"]))), row["llm_score"], "Raw margin to probability")
        groups[cohort] = rows
        means[cohort] = dict(zip(features, matrix.mean(axis=0)))
        for i, feature in enumerate(features):
            record = next(r for r in summary if r["analysis"] == "deepseek_to_llama" and r["group"] == cohort and r["feature"] == feature)
            if int(record["records"]) != len(rows):
                raise ValueError("The saved SHAP summary denominator differs.")
            close(matrix[:, i].mean(), record["mean_signed_shap"], "Signed SHAP mean")
            close(np.abs(matrix[:, i]).mean(), record["mean_abs_shap"], "Absolute SHAP mean")
    selected = sorted(features, key=lambda f: (means["missed_llama"][f], f))[:3]
    if any(means["missed_llama"][f] >= 0 for f in selected):
        raise ValueError("This diagnostic requires three negative mean contributions.")
    plotted = []
    for feature in selected:
        for cohort in ("missed_llama", "detected_llama"):
            plotted.append({"feature": feature, "cohort": cohort, "emails": len(groups[cohort]),
                            "mean_signed_shap": float(means[cohort][feature]), "score_scale": "raw margin (log-odds)"})
    style()
    fig, ax = plt.subplots(figsize=(9.1, 4.8))
    fig.subplots_adjust(left=0.30, right=0.96, bottom=0.28, top=0.69)
    heading(fig, "What pulled CatBoost's score down on missed Llama emails?",
            "Trained on DeepSeek 3.2 / Tested on Llama 3.3 70B / Default threshold 0.5")
    for y, feature in enumerate(selected):
        missed, detected = means["missed_llama"][feature], means["detected_llama"][feature]
        ax.plot([missed, detected], [y, y], color="#b8c4d0", linewidth=2, zorder=1)
        ax.scatter(missed, y, color="#b54720", s=45, zorder=3)
        ax.scatter(detected, y, color="#126782", s=45, marker="D", zorder=3)
        for value, color, side in ((missed, "#b54720", -1), (detected, "#126782", 1)):
            ax.annotate(f"{value:+.3f}", (value, y), xytext=(side * 8, 0), textcoords="offset points",
                        ha="right" if side < 0 else "left", va="center", fontsize=10, color=color, weight="bold")
    maximum = max(abs(means[c][f]) for c in means for f in selected)
    limit = float(np.ceil((maximum + 0.15) * 10) / 10)
    ax.set(xlim=(-limit, limit), ylim=(2.5, -0.5), xlabel="Mean signed SHAP contribution (log-odds)")
    ax.set_yticks(range(3), [FEATURE_NAMES[f] for f in selected])
    ax.axvline(0, color="#7c8794", linewidth=1)
    clean_axes(ax)
    legend = [Line2D([], [], marker="o", linestyle="none", color="#b54720", label=f"Missed ({len(groups['missed_llama'])} emails)"),
              Line2D([], [], marker="D", linestyle="none", color="#126782", label=f"Detected ({len(groups['detected_llama'])} emails)")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.035, 0.83), frameon=False, ncol=2, fontsize=10)
    fig.text(0.30, 0.13, "<-- Toward human-written", color=MUTED, fontsize=9)
    fig.text(0.96, 0.13, "Toward generated -->", ha="right", color=MUTED, fontsize=9)
    fig.text(0.045, 0.045, "Shown are the three features with the most negative mean SHAP contribution among missed emails.", color=MUTED, fontsize=9)
    finish(fig, "shap/catboost/catboost_shap_llama_missed_vs_detected", plotted, inputs,
           "Select the three smallest mean signed contributions among missed Llama emails; compare their means on detected Llama emails. Means are recomputed from verified per-email native CatBoost SHAP contributions. These are associations with model scores, not causal effects.", args.check_only)


if __name__ == "__main__":
    main()
