from __future__ import annotations

import pandas as pd


def summarize_demand(df: pd.DataFrame, value_col: str = "mw_clean") -> dict[str, object]:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    values = pd.to_numeric(data[value_col], errors="coerce")
    peak_idx = values.idxmax()
    valley_idx = values.idxmin()
    return {
        "rows": int(len(data)),
        "start": data["timestamp"].min(),
        "end": data["timestamp"].max(),
        "mean_mw": float(values.mean()),
        "median_mw": float(values.median()),
        "std_mw": float(values.std()),
        "min_mw": float(values.min()),
        "min_timestamp": data.loc[valley_idx, "timestamp"],
        "max_mw": float(values.max()),
        "max_timestamp": data.loc[peak_idx, "timestamp"],
        "missing_values": int(values.isna().sum()),
    }
