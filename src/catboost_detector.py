"""CatBoost detector definition.

Running this file constructs a model without loading research data or training.
"""

import catboost as cb

from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def build_catboost(
    iterations=100,
    depth=6,
    learning_rate=0.05,
):
    """Create a fresh, unfitted CatBoost pipeline."""

    return Pipeline(
        steps=[
            (
                "catboost",
                cb.CatBoostClassifier(
                    loss_function="Logloss",
                    iterations=iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                    l2_leaf_reg=3.0,
                    boosting_type="Plain",
                    grow_policy="SymmetricTree",
                    random_seed=42,
                    thread_count=1,
                    task_type="CPU",
                    use_best_model=False,
                    allow_writing_files=False,
                    verbose=False,
                ),
            ),
        ]
    )


def catboost_parameter_grid():
    """Return proposed settings for development-only selection."""

    return {
        "catboost__iterations": [100, 200],
        "catboost__depth": [3, 4, 5, 6, 7],
        "catboost__learning_rate": [0.05, 0.1],
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

    model = build_catboost()
    candidates = list(ParameterGrid(catboost_parameter_grid()))

    print("CatBoost version:", cb.__version__)
    print("CatBoost pipeline constructed:")
    print(model)

    print(
        "\nExplicit settings:",
        model.named_steps["catboost"].get_params(),
    )
    print("Proposed parameter combinations:", len(candidates))
    print("No research data loaded.")
    print("No model fitted.")
    print("No performance measured.")


if __name__ == "__main__":
    main()