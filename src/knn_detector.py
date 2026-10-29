"""KNN detector definition.

This module does not read datasets, create partitions or train automatically.
"""

from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def build_knn(n_neighbors=5, weights="uniform", p=2):
    """Create a fresh, unfitted scaling-and-KNN pipeline."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm="brute",
                    metric="minkowski",
                    p=p,
                    n_jobs=1,
                ),
            ),
        ]
    )


def knn_parameter_grid():
    """Return proposed settings for development-only model selection."""

    return {
        "knn__n_neighbors": [3, 5, 11, 21, 31],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }


def predict_llm_scores(model, X):
    """Return scores for class 1 from an already fitted pipeline."""

    check_is_fitted(model)

    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(
            "Expected binary labels: 0 = human and 1 = LLM."
        )

    probabilities = model.predict_proba(X)
    llm_column = classes.index(1)

    return probabilities[:, llm_column]


def main():
    """Check construction and display the proposed configuration."""

    model = build_knn()
    candidates = list(ParameterGrid(knn_parameter_grid()))

    print("KNN pipeline constructed:")
    print(model)

    print("\nProposed parameter combinations:", len(candidates))
    print("No research data loaded.")
    print("No model fitted.")
    print("No performance measured.")


if __name__ == "__main__":
    main()