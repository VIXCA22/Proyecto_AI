from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    error = truth - pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    nonzero = np.abs(truth) > 1e-9
    mape = float(np.mean(np.abs(error[nonzero] / truth[nonzero])) * 100) if nonzero.any() else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def metrics_frame(rows: list[dict]) -> pd.DataFrame:
    columns = ["model", "fold", "mae", "rmse", "mape", "train_rows", "test_rows"]
    return pd.DataFrame(rows, columns=columns)
