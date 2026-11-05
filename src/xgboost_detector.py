from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def build_xgboost():
    return Pipeline([("xgboost", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=1,
    ))])


def xgboost_parameter_grid():
    return {
        "xgboost__n_estimators": [100, 200],
        "xgboost__max_depth": [2, 3, 4, 6, 8],
        "xgboost__learning_rate": [0.05, 0.1],
    }