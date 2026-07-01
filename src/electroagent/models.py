from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class HistoricalAverageRegressor:
    """Predice usando el promedio histórico calculado por franja horaria."""

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        values = X["hist_slot_mean"].to_numpy(dtype=float)
        return np.where(np.isnan(values), self.fallback_, values)


def model_registry() -> dict[str, type[HistoricalAverageRegressor]]:
    return {"promedio_historico": HistoricalAverageRegressor}


@dataclass(frozen=True)
class ModelSelection:
    name: str
    estimator: HistoricalAverageRegressor
    metrics: pd.DataFrame
    ranking: pd.DataFrame
