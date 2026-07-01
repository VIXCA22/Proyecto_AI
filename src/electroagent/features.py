from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .context import EXTERNAL_FEATURE_PREFIXES, calendar_features


@dataclass
class FeatureBuilder:
    target_col: str = "mw_clean"
    frequency: str = "15min"
    lags: tuple[int, ...] = (1, 2, 4, 8, 96, 192, 672)
    rolling_windows: tuple[int, ...] = (4, 16, 96)
    feature_names_: list[str] = field(default_factory=list)
    active_lags_: tuple[int, ...] = field(default_factory=tuple)
    external_feature_cols_: list[str] = field(default_factory=list)
    external_means_: dict[str, float] = field(default_factory=dict)
    external_slot_means_: dict[str, dict[int, float]] = field(default_factory=dict)
    global_mean_: float = float("nan")
    slot_means_: dict[tuple[int, int], float] = field(default_factory=dict)
    any_day_slot_means_: dict[int, float] = field(default_factory=dict)

    def _indexed(self, df: pd.DataFrame) -> pd.DataFrame:
        indexed = df.copy()
        indexed["timestamp"] = pd.to_datetime(indexed["timestamp"])
        indexed = indexed.sort_values("timestamp").drop_duplicates("timestamp")
        return indexed.set_index("timestamp")

    @staticmethod
    def _slot(index: pd.DatetimeIndex) -> pd.Index:
        return index.hour * 4 + (index.minute // 15)

    @staticmethod
    def _time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
        hour_float = index.hour + index.minute / 60.0
        slot = index.hour * 4 + (index.minute // 15)
        features = pd.DataFrame(index=index)
        features["slot"] = slot
        features["is_weekend"] = (index.dayofweek >= 5).astype(int)
        features["hour_sin"] = np.sin(2 * np.pi * hour_float / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour_float / 24)
        features["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
        features["doy_sin"] = np.sin(2 * np.pi * index.dayofyear / 366)
        features["doy_cos"] = np.cos(2 * np.pi * index.dayofyear / 366)
        return features

    @staticmethod
    def _external_columns(df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in df.columns
            if any(str(column).startswith(prefix) for prefix in EXTERNAL_FEATURE_PREFIXES)
            and pd.api.types.is_numeric_dtype(df[column])
            and pd.to_numeric(df[column], errors="coerce").notna().any()
        ]

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        indexed = self._indexed(df)
        target = pd.to_numeric(indexed[self.target_col], errors="coerce")
        self.global_mean_ = float(target.mean())
        self.active_lags_ = tuple(lag for lag in self.lags if lag < len(target))
        self.external_feature_cols_ = self._external_columns(indexed)

        slot = self._slot(indexed.index)
        profile = pd.DataFrame(
            {"y": target, "dayofweek": indexed.index.dayofweek, "slot": slot},
            index=indexed.index,
        )
        self.slot_means_ = profile.groupby(["dayofweek", "slot"])["y"].mean().to_dict()
        self.any_day_slot_means_ = profile.groupby("slot")["y"].mean().to_dict()

        features = self._time_features(indexed.index)
        features = features.join(calendar_features(indexed.index))
        for column in self.external_feature_cols_:
            values = pd.to_numeric(indexed[column], errors="coerce").ffill().bfill()
            self.external_means_[column] = float(values.mean())
            self.external_slot_means_[column] = values.groupby(slot).mean().to_dict()
            features[column] = values.fillna(self.external_means_[column])

        for lag in self.active_lags_:
            features[f"lag_{lag}"] = target.shift(lag)

        shifted = target.shift(1)
        for window in self.rolling_windows:
            min_periods = max(2, min(window, window // 4))
            features[f"rolling_mean_{window}"] = shifted.rolling(window, min_periods=min_periods).mean()
            features[f"rolling_std_{window}"] = shifted.rolling(window, min_periods=min_periods).std().fillna(0)

        hist_same_slot = profile.groupby(["dayofweek", "slot"])["y"].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
        hist_any_day = profile.groupby("slot")["y"].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
        features["hist_slot_mean"] = hist_same_slot.fillna(hist_any_day).fillna(shifted.expanding().mean())

        supervised = features.assign(y=target).dropna()
        self.feature_names_ = [column for column in supervised.columns if column != "y"]
        return supervised[self.feature_names_], supervised["y"]

    def _external_values(
        self,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
        future_context: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        slot = int(self._slot(pd.DatetimeIndex([timestamp]))[0])
        context = None
        if future_context is not None and not future_context.empty:
            context = future_context.copy()
            context["timestamp"] = pd.to_datetime(context["timestamp"])
            matches = context[context["timestamp"] == timestamp]
            if matches.empty:
                nearest = pd.merge_asof(
                    pd.DataFrame({"timestamp": [timestamp]}),
                    context.sort_values("timestamp"),
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.to_timedelta(self.frequency),
                )
                matches = nearest.dropna(axis=1, how="all")
            if not matches.empty:
                context = matches.iloc[0]

        for column in self.external_feature_cols_:
            value = np.nan
            if context is not None and column in context.index:
                value = pd.to_numeric(pd.Series([context[column]]), errors="coerce").iloc[0]
            if pd.isna(value) and column in history.columns:
                previous = pd.to_numeric(history[column], errors="coerce").dropna()
                if not previous.empty:
                    value = float(previous.iloc[-1])
            if pd.isna(value):
                value = self.external_slot_means_.get(column, {}).get(slot, self.external_means_.get(column, 0.0))
            values[column] = float(value)
        return values

    def transform_next(
        self,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
        future_context: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("FeatureBuilder must be fitted before forecasting")

        indexed = self._indexed(history)
        target = pd.to_numeric(indexed[self.target_col], errors="coerce").dropna()
        index = pd.DatetimeIndex([pd.Timestamp(timestamp)])
        features = self._time_features(index)
        features = features.join(calendar_features(index))
        for column, value in self._external_values(history, pd.Timestamp(timestamp), future_context).items():
            features[column] = value

        for lag in self.active_lags_:
            features[f"lag_{lag}"] = float(target.iloc[-lag]) if len(target) >= lag else self.global_mean_

        for window in self.rolling_windows:
            tail = target.iloc[-window:]
            features[f"rolling_mean_{window}"] = float(tail.mean()) if len(tail) else self.global_mean_
            std = float(tail.std()) if len(tail) > 1 else 0.0
            features[f"rolling_std_{window}"] = 0.0 if np.isnan(std) else std

        dow = int(index.dayofweek[0])
        slot = int(self._slot(index)[0])
        features["hist_slot_mean"] = self.slot_means_.get(
            (dow, slot),
            self.any_day_slot_means_.get(slot, self.global_mean_),
        )
        return features[self.feature_names_]
