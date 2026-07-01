from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningReport:
    rows_in: int
    rows_out: int
    missing_timestamps: int
    missing_values: int
    detected_anomalies: int
    corrected_points: int
    frequency: str


def regularize_frequency(df: pd.DataFrame, frequency: str = "15min") -> tuple[pd.DataFrame, int]:
    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp")
    indexed = ordered.set_index("timestamp")
    full_index = pd.date_range(indexed.index.min(), indexed.index.max(), freq=frequency)
    regularized = indexed.reindex(full_index)
    regularized.index.name = "timestamp"
    return regularized.reset_index(), len(full_index) - len(indexed)


def detect_rolling_mad_anomalies(
    series: pd.Series,
    window: int = 96,
    threshold: float = 5.0,
    min_positive: float = 1.0,
    min_delta_mw: float = 40.0,
    relative_delta: float = 0.08,
) -> pd.Series:
    """Detecta puntos anómalos comparando cada valor con su entorno."""
    values = pd.to_numeric(series, errors="coerce")
    rolling_median = values.rolling(window=window, center=True, min_periods=max(8, window // 4)).median()
    residual = (values - rolling_median).abs()
    rolling_mad = residual.rolling(window=window, center=True, min_periods=max(8, window // 4)).median()
    robust_z = 0.6745 * residual / rolling_mad.replace(0, np.nan)

    physical_error = values <= min_positive
    statistical_error = robust_z > threshold
    flat_profile_error = (rolling_mad.fillna(0) <= 1e-9) & (
        residual > (rolling_median.abs() * relative_delta).clip(lower=min_delta_mw)
    )
    return (physical_error | statistical_error | flat_profile_error).fillna(False)


def clean_demand(
    df: pd.DataFrame,
    frequency: str = "15min",
    target_col: str = "mw",
    anomaly_window: int = 96,
    anomaly_threshold: float = 5.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Ordena la serie, marca anomalías y corrige la demanda."""
    rows_in = len(df)
    regularized, missing_timestamps = regularize_frequency(df, frequency=frequency)
    values = pd.to_numeric(regularized[target_col], errors="coerce")
    missing_values = int(values.isna().sum())
    anomalies = detect_rolling_mad_anomalies(values, window=anomaly_window, threshold=anomaly_threshold)

    cleaned = regularized.copy()
    cleaned["is_anomaly"] = anomalies
    cleaned["mw_original"] = values
    cleaned["mw_clean"] = values.mask(anomalies)
    cleaned["mw_clean"] = cleaned["mw_clean"].interpolate(method="linear", limit_direction="both")
    cleaned["mw_clean"] = cleaned["mw_clean"].ffill().bfill()

    corrected_points = int(anomalies.sum() + missing_values)
    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(cleaned),
        missing_timestamps=missing_timestamps,
        missing_values=missing_values,
        detected_anomalies=int(anomalies.sum()),
        corrected_points=corrected_points,
        frequency=frequency,
    )
    return cleaned.reset_index(drop=True), report
