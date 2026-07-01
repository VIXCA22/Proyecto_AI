import numpy as np
import pandas as pd

from electroagent.agente import AgentePrediccion


def synthetic_demand(days: int = 21) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=days * 96, freq="15min")
    hour = timestamps.hour + timestamps.minute / 60
    daily = 160 * np.sin(2 * np.pi * (hour - 7) / 24)
    weekly = np.where(timestamps.dayofweek >= 5, -70, 40)
    trend = np.linspace(0, 25, len(timestamps))
    values = 1350 + daily + weekly + trend
    return pd.DataFrame({"timestamp": timestamps, "mw": values})


def test_agente_returns_forecast_and_ranking():
    df = synthetic_demand()
    agent = AgentePrediccion(validation_splits=2, validation_test_size=48)

    result = agent.run(df, horizon_steps=8)

    assert len(result.forecast) == 8
    assert not result.validation_comparison.empty
    assert {"timestamp", "cence_mw", "prediction_mw"}.issubset(result.validation_comparison.columns)
    assert result.selection.name in set(result.selection.ranking["model"])
    assert result.forecast["prediction_mw"].notna().all()
