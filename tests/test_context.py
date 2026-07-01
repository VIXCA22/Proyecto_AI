import pandas as pd

from electroagent.context import calendar_features, normalize_weather_frame


def test_calendar_features_marks_costa_rica_event():
    index = pd.DatetimeIndex(["2026-09-15 12:00:00", "2026-09-16 12:00:00"])

    features = calendar_features(index)

    assert features.loc[index[0], "calendar_is_cr_holiday"] == 1
    assert features.loc[index[1], "event_is_window"] == 1


def test_normalize_imn_weather_table_values():
    raw = pd.DataFrame(
        {
            "Fecha": ["29/06/2026 11:00 p.m."],
            "Temp": ["25,29"],
            "Lluvia": ["0,00"],
            "Rad_max": ["0,00"],
            "Pres_atm": ["1.011,40"],
        }
    )

    weather = normalize_weather_frame(raw)

    assert weather.loc[0, "timestamp"] == pd.Timestamp("2026-06-29 23:00:00")
    assert weather.loc[0, "weather_temp_c"] == 25.29
    assert weather.loc[0, "weather_pressure_hpa"] == 1011.40
