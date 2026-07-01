from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

import warnings

from .cleaning import CleaningReport, clean_demand
from .context import enrich_demand_context
from .features import FeatureBuilder
from .metrics import metrics_frame, regression_metrics
from .models import ModelSelection, model_registry
from .stats import summarize_demand


@dataclass(frozen=True)
class AgentResult:
    selection: ModelSelection
    forecast: pd.DataFrame
    cleaned: pd.DataFrame
    cleaning_report: CleaningReport
    summary: dict[str, object]
    context_columns: list[str]


class ForecastAgent:
    """Orchestrates tool selection for electrical demand forecasting."""

    def __init__(
        self,
        frequency: str = "15min",
        validation_splits: int = 3,
        validation_test_size: int = 96,
    ):
        self.frequency = frequency
        self.validation_splits = validation_splits
        self.validation_test_size = validation_test_size

    def run(
        self,
        df: pd.DataFrame,
        horizon_steps: int = 96,
        weather_df: pd.DataFrame | None = None,
        enos_phase: str | None = None,
    ) -> AgentResult:
        contextualized = enrich_demand_context(df, weather_df=weather_df, enos_phase=enos_phase, frequency=self.frequency)
        cleaned, cleaning_report = clean_demand(contextualized, frequency=self.frequency)
        cleaned = enrich_demand_context(cleaned, weather_df=None, enos_phase=enos_phase, frequency=self.frequency)
        future_context = self._future_context(cleaned, horizon_steps=horizon_steps, weather_df=weather_df, enos_phase=enos_phase)
        selection, builder = self._select_by_recursive_backtest(cleaned, horizon_steps=horizon_steps)
        forecast = self._forecast_recursive(
            cleaned,
            builder,
            selection,
            horizon_steps=horizon_steps,
            future_context=future_context,
        )
        summary = summarize_demand(cleaned)
        context_columns = list(builder.external_feature_cols_)
        return AgentResult(
            selection=selection,
            forecast=forecast,
            cleaned=cleaned,
            cleaning_report=cleaning_report,
            summary=summary,
            context_columns=context_columns,
        )

    def _future_context(
        self,
        cleaned: pd.DataFrame,
        horizon_steps: int,
        weather_df: pd.DataFrame | None,
        enos_phase: str | None,
    ) -> pd.DataFrame:
        last_timestamp = pd.to_datetime(cleaned["timestamp"]).max()
        step = pd.to_timedelta(self.frequency)
        future_index = pd.date_range(last_timestamp + step, periods=horizon_steps, freq=self.frequency)
        future = pd.DataFrame({"timestamp": future_index})
        return enrich_demand_context(future, weather_df=weather_df, enos_phase=enos_phase, frequency=self.frequency)

    def _select_by_recursive_backtest(
        self,
        cleaned: pd.DataFrame,
        horizon_steps: int,
    ) -> tuple[ModelSelection, FeatureBuilder]:
        registry = model_registry()
        horizon = min(horizon_steps, self.validation_test_size)
        configured_lags = FeatureBuilder().lags
        rows: list[dict] = []

        n_rows = len(cleaned)
        active_lags = tuple(lag for lag in configured_lags if lag < n_rows)
        min_train_rows = max(30, (max(active_lags) if active_lags else 1) + horizon)
        origins = []
        for split in range(self.validation_splits, 0, -1):
            train_end = n_rows - split * horizon
            test_end = train_end + horizon
            if train_end >= min_train_rows and test_end <= n_rows:
                origins.append((train_end, test_end))

        if not origins:
            raise ValueError("not enough rows for recursive temporal validation")

        for model_name, estimator in registry.items():
            for fold, (train_end, test_end) in enumerate(origins, start=1):
                train_df = cleaned.iloc[:train_end].copy()
                test_df = cleaned.iloc[train_end:test_end].copy()
                builder = FeatureBuilder(frequency=self.frequency)
                X_train, y_train = builder.fit_transform(train_df)
                fitted = clone(estimator)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    fitted.fit(X_train, y_train)
                selection = ModelSelection(
                    name=model_name,
                    estimator=fitted,
                    metrics=pd.DataFrame(),
                    ranking=pd.DataFrame(),
                )
                forecast = self._forecast_recursive(
                    train_df,
                    builder,
                    selection,
                    horizon_steps=len(test_df),
                    future_context=test_df,
                )
                metric_values = regression_metrics(test_df["mw_clean"], forecast["prediction_mw"])
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        **metric_values,
                        "train_rows": len(train_df),
                        "test_rows": len(test_df),
                    }
                )

        metrics = metrics_frame(rows)
        ranking = (
            metrics.groupby("model", as_index=False)[["mae", "rmse", "mape"]]
            .mean()
            .sort_values(["mae", "rmse"], ascending=True)
            .reset_index(drop=True)
        )
        best_name = str(ranking.iloc[0]["model"])

        final_builder = FeatureBuilder(frequency=self.frequency)
        X_full, y_full = final_builder.fit_transform(cleaned)
        final_estimator = clone(registry[best_name])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            final_estimator.fit(X_full, y_full)

        return (
            ModelSelection(name=best_name, estimator=final_estimator, metrics=metrics, ranking=ranking),
            final_builder,
        )

    def _forecast_recursive(
        self,
        cleaned: pd.DataFrame,
        builder: FeatureBuilder,
        selection: ModelSelection,
        horizon_steps: int,
        future_context: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        history_cols = ["timestamp", "mw_clean"] + [
            column for column in builder.external_feature_cols_ if column in cleaned.columns
        ]
        history = cleaned[history_cols].copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history = history.sort_values("timestamp")
        step = pd.to_timedelta(self.frequency)
        rows: list[dict] = []

        for horizon in range(1, horizon_steps + 1):
            next_timestamp = history["timestamp"].max() + step
            X_next = builder.transform_next(history, next_timestamp, future_context=future_context)
            prediction = float(selection.estimator.predict(X_next)[0])
            rows.append(
                {
                    "timestamp": next_timestamp,
                    "prediction_mw": prediction,
                    "horizon_step": horizon,
                    "model": selection.name,
                }
            )
            new_history_row = {"timestamp": next_timestamp, "mw_clean": prediction}
            for column in builder.external_feature_cols_:
                if column in X_next.columns:
                    new_history_row[column] = float(X_next.iloc[0][column])
            history = pd.concat([history, pd.DataFrame([new_history_row])], ignore_index=True)

        return pd.DataFrame(rows)
