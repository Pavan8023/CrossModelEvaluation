# Reproducible paper figures

These are reconstructed, saved replacements for the earlier one-off Python
plotting commands. They use the same experiment results and reproduce the
figure designs; they are not presented as the recovered historical source.

Each figure has its own script. The shared `plot_utils.py` handles verified
input loading, common appearance, and output/provenance files.

## Run from the project root

Use the existing project environment. The plotting scripts need Python,
NumPy, and Matplotlib; they do not need the classifiers or spaCy to run.

```powershell
.\.venv\Scripts\python.exe .\src\plots\plot_intra_model_f1.py
.\.venv\Scripts\python.exe .\src\plots\plot_cross_model_f1.py
.\.venv\Scripts\python.exe .\src\plots\plot_recalibration_errors.py
.\.venv\Scripts\python.exe .\src\plots\plot_cross_dataset_fpr.py
.\.venv\Scripts\python.exe .\src\plots\plot_aggregated_pool_change.py
.\.venv\Scripts\python.exe .\src\plots\plot_catboost_shap_errors.py
```

Append `--check-only` to any command to validate inputs and calculations
without writing figure files. No script trains models, tunes parameters,
selects decision thresholds, generates emails, or calls an API.

## Imports and where they are used

| Import | Use in these scripts |
|---|---|
| `matplotlib.pyplot` | Creates axes, error bars, heatmaps, bar charts, legends, and image exports. Initialized in `plot_utils.py` with the noninteractive Agg backend; the scripts import its `plt` alias. |
| `numpy` | Arrays, fold means/sample SD, source-rate averages, pooled F1 differences, and SHAP contribution means. Imported in the helper and used through its `np` alias. |
| `matplotlib.colors.Normalize` | Gives every cross-model heatmap the same numeric color scale. |
| `matplotlib.lines.Line2D`, `matplotlib.patches.Patch` | Build marker and color legends in the error, source, and SHAP figures. |
| `csv`, `json` | Read saved result tables, ledgers, and the pinned source plan; write plotted values and provenance. |
| `hashlib`, `pathlib` | Check source/output SHA-256 hashes and resolve paths relative to this project. |
| `argparse` | Provides the `--check-only` command-line option. |
| `math`, `platform` | Compare numeric values with a tolerance and record the Python version. |

Only NumPy and Matplotlib are third-party plotting dependencies. The other
utility modules above belong to Python's standard library. CatBoost SHAP
calculation itself remains in `src/explain_catboost_shap.py`; the new SHAP
plotter reads and verifies its saved outputs.

## Scripts and outputs

All output paths below are relative to the existing project root.

| Script | Output stem under `figures/` | Calculation |
|---|---|---|
| `plot_intra_model_f1.py` | `intra_model/summary/intra_model_f1_five_classifiers` | Five outer-fold F1 scores: mean and sample SD (`ddof=1`), not a confidence interval. |
| `plot_cross_model_f1.py` | `cross_model/cross_model_f1_before_after_recalibration` | Paired 3-by-3 F1 matrices for each classifier, before/after threshold adjustment. |
| `plot_recalibration_errors.py` | `recalibration/deepseek_to_llama_threshold_tradeoff` | Default and adjusted FN/FP counts on identical DeepSeek-to-Llama test records. |
| `plot_cross_dataset_fpr.py` | `cross_dataset/human_source_false_positive_rates` | Held-out-source human FPR versus the mean of the four models including that source. |
| `plot_aggregated_pool_change.py` | `aggregated_pool/pooled_training_f1_change` | Per-target pooled F1 minus the mean F1 of the two models trained on other generators, multiplied by 100. |
| `plot_catboost_shap_errors.py` | `shap/catboost/catboost_shap_llama_missed_vs_detected` | Three most negative signed SHAP means among misses, compared with the corresponding means among detections. |

Every script writes PNG (450 dpi), SVG, full-precision `.data.csv`, and
`.sources.json` files next to the existing figure. Re-running overwrites only
those figure-related outputs; original datasets and experimental results
are read-only. The intra-model plot is available in the project even though
the current manuscript presents those scores in a table instead of an image.

## Traceability for supervisor review

1. `figure_sources.json` pins the exact ledger files and their SHA-256 hashes.
   No script silently switches to the latest experiment.
2. Each CSV is checked against the output hash in its experiment ledger.
   Calculations include consistency checks against confusion counts, saved
   means, source comparisons, or per-email SHAP values where appropriate.
3. Each output's `.sources.json` records the plotting script, shared helper,
   source plan, source ledgers, input CSV hashes, experiment identifiers,
   software versions, output hashes, and the calculation used.
4. The `.data.csv` exposes the exact unrounded values used by the plot.
   Displayed labels can be rounded without changing those saved values.
5. Figure styles and dimensions are set in code. SVG timestamps are omitted
   and the SVG identifier salt is fixed for repeatable output in the same
   software/font environment. Different library or font versions can change
   rendering; the source values remain checkable.

Keep the source plan and shared helper with the six scripts when sharing
them. The corresponding `ledger/` and `results/` files are also required.
Existing old source records are preserved for historical traceability.
