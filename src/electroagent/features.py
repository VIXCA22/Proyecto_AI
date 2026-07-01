from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureBuilder:
    target_col: str = "mw_clean"
    frequency: str = "15min"
    feature_names_: list[str] = field(default_factory=list)
    global_mean_: float = float("nan")
    slot_means_: dict[tuple[int, int], float] = field(default_factory=dict)
    any_day_slot_means_: dict[int, float] = field(default_factory=dict)

    def _indexed(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp").drop_duplicates("timestamp")
        return data.set_index("timestamp")

    @staticmethod
    def _slot(index: pd.DatetimeIndex) -> pd.Index:
        return index.hour * 4 + (index.minute // 15)

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        indexed = self._indexed(df)
        target = pd.to_numeric(indexed[self.target_col], errors="coerce")
        slot = self._slot(indexed.index)
        profile = pd.DataFrame(
            {"y": target, "dayofweek": indexed.index.dayofweek, "slot": slot},
            index=indexed.index,
        )

        self.global_mean_ = float(target.mean())
        self.slot_means_ = profile.groupby(["dayofweek", "slot"])["y"].mean().to_dict()
        self.any_day_slot_means_ = profile.groupby("slot")["y"].mean().to_dict()

        hist_same_slot = profile.groupby(["dayofweek", "slot"])["y"].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
        hist_any_day = profile.groupby("slot")["y"].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
        features = pd.DataFrame(index=indexed.index)
        features["hist_slot_mean"] = hist_same_slot.fillna(hist_any_day).fillna(target.shift(1).expanding().mean())

        supervised = features.assign(y=target).dropna()
        self.feature_names_ = ["hist_slot_mean"]
        return supervised[self.feature_names_], supervised["y"]

    def transform_next(self, history: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("primero se debe ajustar FeatureBuilder")

        index = pd.DatetimeIndex([pd.Timestamp(timestamp)])
        dow = int(index.dayofweek[0])
        slot = int(self._slot(index)[0])
        hist_slot_mean = self.slot_means_.get(
            (dow, slot),
            self.any_day_slot_means_.get(slot, self.global_mean_),
        )
        return pd.DataFrame({"hist_slot_mean": [hist_slot_mean]}, index=index)
