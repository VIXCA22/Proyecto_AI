from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cleaning import CleaningReport, clean_demand
from .features import FeatureBuilder
from .metrics import metrics_frame, regression_metrics
from .models import ModelSelection, model_registry
from .stats import summarize_demand


@dataclass(frozen=True)
class ResultadoAgente:
    selection: ModelSelection
    forecast: pd.DataFrame
    validation_comparison: pd.DataFrame
    cleaned: pd.DataFrame
    cleaning_report: CleaningReport
    summary: dict[str, object]


class AgentePrediccion:
    """Orquesta la limpieza, la validación y el pronóstico de demanda."""

    def __init__(
        self,
        frequency: str = "15min",
        validation_splits: int = 3,
        validation_test_size: int = 96,
    ):
        self.frequency = frequency
        self.validation_splits = validation_splits
        self.validation_test_size = validation_test_size

    def run(self, df: pd.DataFrame, horizon_steps: int = 96) -> ResultadoAgente:
        cleaned, cleaning_report = clean_demand(df, frequency=self.frequency)
        selection, builder, validation_comparison = self._validate_historical_average(
            cleaned,
            horizon_steps=horizon_steps,
        )
        forecast = self._forecast_recursive(
            cleaned,
            builder,
            selection,
            horizon_steps=horizon_steps,
        )
        return ResultadoAgente(
            selection=selection,
            forecast=forecast,
            validation_comparison=validation_comparison,
            cleaned=cleaned,
            cleaning_report=cleaning_report,
            summary=summarize_demand(cleaned),
        )

    def _validate_historical_average(
        self,
        cleaned: pd.DataFrame,
        horizon_steps: int,
    ) -> tuple[ModelSelection, FeatureBuilder, pd.DataFrame]:
        registry = model_registry()
        model_name, model_class = next(iter(registry.items()))
        horizon = min(horizon_steps, self.validation_test_size)
        rows: list[dict] = []
        comparison_rows: list[dict] = []

        n_rows = len(cleaned)
        min_train_rows = max(30, horizon + 1)
        origins: list[tuple[int, int]] = []
        for split in range(self.validation_splits, 0, -1):
            train_end = n_rows - split * horizon
            test_end = train_end + horizon
            if train_end >= min_train_rows and test_end <= n_rows:
                origins.append((train_end, test_end))

        if not origins:
            raise ValueError("no hay suficientes datos para hacer la validación")

        for fold, (train_end, test_end) in enumerate(origins, start=1):
            train_df = cleaned.iloc[:train_end].copy()
            test_df = cleaned.iloc[train_end:test_end].copy()
            builder = FeatureBuilder(frequency=self.frequency)
            X_train, y_train = builder.fit_transform(train_df)
            fitted = model_class()
            fitted.fit(X_train, y_train)

            validation_selection = ModelSelection(
                name=model_name,
                estimator=fitted,
                metrics=pd.DataFrame(),
                ranking=pd.DataFrame(),
            )
            validation_forecast = self._forecast_recursive(
                train_df,
                builder,
                validation_selection,
                horizon_steps=len(test_df),
            )
            metric_values = regression_metrics(test_df["mw_clean"], validation_forecast["prediction_mw"])
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    **metric_values,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                }
            )
            comparison_rows.extend(
                self._comparison_rows(test_df, validation_forecast, fold=fold, model_name=model_name)
            )

        metrics = metrics_frame(rows)
        ranking = (
            metrics.groupby("model", as_index=False)[["mae", "rmse", "mape"]]
            .mean()
            .sort_values(["mae", "rmse"], ascending=True)
            .reset_index(drop=True)
        )

        final_builder = FeatureBuilder(frequency=self.frequency)
        X_full, y_full = final_builder.fit_transform(cleaned)
        final_estimator = model_class()
        final_estimator.fit(X_full, y_full)

        selection = ModelSelection(
            name=model_name,
            estimator=final_estimator,
            metrics=metrics,
            ranking=ranking,
        )
        return selection, final_builder, pd.DataFrame(comparison_rows)

    def _comparison_rows(
        self,
        test_df: pd.DataFrame,
        forecast: pd.DataFrame,
        fold: int,
        model_name: str,
    ) -> list[dict]:
        comparison = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(test_df["timestamp"]).to_numpy(),
                "cence_mw": pd.to_numeric(test_df["mw_clean"], errors="coerce").to_numpy(),
                "prediction_mw": pd.to_numeric(forecast["prediction_mw"], errors="coerce").to_numpy(),
                "fold": fold,
                "model": model_name,
            }
        )
        if "mw_programmed" in test_df.columns:
            comparison["cence_programmed_mw"] = pd.to_numeric(test_df["mw_programmed"], errors="coerce").to_numpy()
        else:
            comparison["cence_programmed_mw"] = pd.NA
        comparison["error_mw"] = comparison["prediction_mw"] - comparison["cence_mw"]
        return comparison.to_dict("records")

    def _forecast_recursive(
        self,
        cleaned: pd.DataFrame,
        builder: FeatureBuilder,
        selection: ModelSelection,
        horizon_steps: int,
    ) -> pd.DataFrame:
        history = cleaned[["timestamp", "mw_clean"]].copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history = history.sort_values("timestamp")
        step = pd.to_timedelta(self.frequency)
        rows: list[dict] = []

        for horizon in range(1, horizon_steps + 1):
            next_timestamp = history["timestamp"].max() + step
            X_next = builder.transform_next(history, next_timestamp)
            prediction = float(selection.estimator.predict(X_next)[0])
            rows.append(
                {
                    "timestamp": next_timestamp,
                    "prediction_mw": prediction,
                    "horizon_step": horizon,
                    "model": selection.name,
                }
            )
            history = pd.concat(
                [
                    history,
                    pd.DataFrame([{"timestamp": next_timestamp, "mw_clean": prediction}]),
                ],
                ignore_index=True,
            )

        return pd.DataFrame(rows)
