from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_random_forest():
    return Pipeline([("random_forest", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    ))])


def random_forest_parameter_grid():
    return {
        "random_forest__n_estimators": [100, 200],
        "random_forest__max_depth": [4, 8, 12, 16, None],
        "random_forest__min_samples_leaf": [1, 4],
    }