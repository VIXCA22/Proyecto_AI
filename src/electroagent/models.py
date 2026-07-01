from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

import warnings

from .metrics import metrics_frame, regression_metrics


class HistoricalAverageRegressor(BaseEstimator, RegressorMixin):
    """Baseline estimator that predicts the historical profile feature."""

    def __init__(self, feature_name: str = "hist_slot_mean"):
        self.feature_name = feature_name

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        if hasattr(X, "columns") and self.feature_name in X.columns:
            self.feature_index_ = list(X.columns).index(self.feature_name)
        else:
            self.feature_index_ = 0
        return self

    def predict(self, X):
        if hasattr(X, "columns") and self.feature_name in X.columns:
            values = X[self.feature_name].to_numpy(dtype=float)
        else:
            values = np.asarray(X)[:, self.feature_index_].astype(float)
        return np.where(np.isnan(values), self.fallback_, values)


def model_registry(random_state: int = 42) -> dict[str, BaseEstimator]:
    return {
        "historical_average": HistoricalAverageRegressor(),
        "knn": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=12, weights="distance")),
            ]
        ),
        "svr": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LinearSVR(C=1.0, epsilon=0.03, random_state=random_state, max_iter=20000)),
                ]
            ),
            transformer=StandardScaler(),
        ),
        "decision_tree": DecisionTreeRegressor(
            max_depth=14,
            min_samples_leaf=8,
            random_state=random_state,
        ),
    }


@dataclass(frozen=True)
class ModelSelection:
    name: str
    estimator: BaseEstimator
    metrics: pd.DataFrame
    ranking: pd.DataFrame


def _safe_splitter(n_rows: int, requested_splits: int, requested_test_size: int) -> TimeSeriesSplit:
    if n_rows < 30:
        raise ValueError("at least 30 supervised rows are required for temporal validation")
    max_test_size = max(1, n_rows // (requested_splits + 2))
    test_size = min(requested_test_size, max_test_size)
    n_splits = min(requested_splits, max(2, (n_rows // test_size) - 1))
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size)


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, BaseEstimator] | None = None,
    n_splits: int = 3,
    test_size: int = 96,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = models or model_registry()
    splitter = _safe_splitter(len(X), n_splits, test_size)
    rows: list[dict] = []

    for model_name, estimator in registry.items():
        for fold, (train_index, test_index) in enumerate(splitter.split(X), start=1):
            fitted = clone(estimator)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fitted.fit(X_train, y_train)
            pred = fitted.predict(X_test)
            metric_values = regression_metrics(y_test, pred)
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    **metric_values,
                    "train_rows": len(train_index),
                    "test_rows": len(test_index),
                }
            )

    metrics = metrics_frame(rows)
    ranking = (
        metrics.groupby("model", as_index=False)[["mae", "rmse", "mape"]]
        .mean()
        .sort_values(["mae", "rmse"], ascending=True)
        .reset_index(drop=True)
    )
    return metrics, ranking


def select_and_fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, BaseEstimator] | None = None,
    n_splits: int = 3,
    test_size: int = 96,
) -> ModelSelection:
    registry = models or model_registry()
    metrics, ranking = evaluate_models(X, y, registry, n_splits=n_splits, test_size=test_size)
    best_name = str(ranking.iloc[0]["model"])
    estimator = clone(registry[best_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(X, y)
    return ModelSelection(name=best_name, estimator=estimator, metrics=metrics, ranking=ranking)
