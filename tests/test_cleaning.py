import pandas as pd

from electroagent.cleaning import clean_demand


def test_clean_demand_detects_and_corrects_spike():
    timestamps = pd.date_range("2025-01-01", periods=200, freq="15min")
    values = pd.Series(1200.0, index=timestamps)
    values.iloc[100] = 5000.0
    df = pd.DataFrame({"timestamp": timestamps, "mw": values.values})

    cleaned, report = clean_demand(df)

    assert report.detected_anomalies >= 1
    assert cleaned.loc[100, "is_anomaly"]
    assert cleaned.loc[100, "mw_clean"] < 1300
