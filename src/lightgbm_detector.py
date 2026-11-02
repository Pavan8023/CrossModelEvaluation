"""LightGBM detector definition.

Running this file constructs a model without loading research data or training.
"""

import lightgbm as lgb

from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def build_lightgbm(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.05,
):
    """Create a fresh, unfitted LightGBM pipeline."""

    return Pipeline(
        steps=[
            (
                "lightgbm",
                lgb.LGBMClassifier(
                    objective="binary",
                    boosting_type="gbdt",
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    max_depth=-1,
                    min_child_samples=20,
                    class_weight=None,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    random_state=42,
                    n_jobs=1,
                    device_type="cpu",
                    deterministic=True,
                    force_col_wise=True,
                    verbosity=-1,
                ),
            ),
        ]
    )


def lightgbm_parameter_grid():
    """Return proposed settings for development-only selection."""

    return {
        "lightgbm__n_estimators": [100, 200],
        "lightgbm__num_leaves": [7, 15, 31, 63, 127],
        "lightgbm__learning_rate": [0.05, 0.1],
    }


def predict_llm_scores(model, X):
    """Return class-1 scores from an already fitted pipeline."""

    check_is_fitted(model)

    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(
            "Expected binary labels: 0 = human and 1 = LLM."
        )

    probabilities = model.predict_proba(X)
    return probabilities[:, classes.index(1)]


def main():
    """Check construction without fitting a model."""

    model = build_lightgbm()
    candidates = list(ParameterGrid(lightgbm_parameter_grid()))

    print("LightGBM version:", lgb.__version__)
    print("LightGBM pipeline constructed:")
    print(model)
    print("\nProposed parameter combinations:", len(candidates))
    print("No research data loaded.")
    print("No model fitted.")
    print("No performance measured.")


if __name__ == "__main__":
    main()