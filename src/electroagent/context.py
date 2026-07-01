from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from html import unescape
from pathlib import Path
from urllib.parse import urljoin
import json
import re

import pandas as pd
import requests


IMN_ENOS_URL = "https://www.imn.ac.cr/boletin-enos"

IMN_STATION_TABLES = {
    "pavas": "https://www.imn.ac.cr/especial/tablas/pave.html",
    "limon": "https://www.imn.ac.cr/especial/tablas/aeroplimon.html",
}

EXTERNAL_FEATURE_PREFIXES = ("weather_", "enos_", "calendar_", "event_")


@dataclass(frozen=True)
class ENOSBulletin:
    year: int | None
    title: str
    description: str
    url: str
    phase: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def infer_enos_phase(text: str) -> str:
    normalized = text.lower()
    if "niño" in normalized or "nino" in normalized:
        return "el_nino"
    if "niña" in normalized or "nina" in normalized:
        return "la_nina"
    if "neutral" in normalized:
        return "neutral"
    return "unknown"


def fetch_latest_enos_bulletin(url: str = IMN_ENOS_URL, timeout: int = 30) -> ENOSBulletin:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    html = response.text

    cards = re.findall(r'<div class="col-md-4 mt-20 boletin-nuevo".*?</div>\s*</div>\s*</div>', html, flags=re.S)
    if not cards:
        cards = re.findall(r'<div class="col-md-4 mt-20 boletin-nuevo".*?</div>', html, flags=re.S)

    bulletins: list[ENOSBulletin] = []
    for card in cards:
        year_match = re.search(r'<h5 class="fecha-boletines">([^<]+)</h5>', card)
        link_match = re.search(r'<a href="([^"]+)"[^>]*>.*?<h2 class="titulo-boletines">([^<]+)</h2>', card, flags=re.S)
        desc_match = re.search(r'<h3 class="descripcion-boletines">(.*?)</h3>', card, flags=re.S)
        if not link_match:
            continue
        year_text = _strip_tags(year_match.group(1)) if year_match else ""
        title = _strip_tags(link_match.group(2))
        description = _strip_tags(desc_match.group(1)) if desc_match else ""
        full_url = urljoin(url, unescape(link_match.group(1)))
        try:
            year = int(year_text)
        except ValueError:
            year = None
        phase = infer_enos_phase(f"{title} {description}")
        bulletins.append(ENOSBulletin(year=year, title=title, description=description, url=full_url, phase=phase))

    if not bulletins:
        raise ValueError("Could not find ENOS bulletins on IMN page")

    return sorted(bulletins, key=lambda item: (item.year or 0, item.title), reverse=True)[0]


def write_enos_bulletin(bulletin: ENOSBulletin, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bulletin.to_json(), ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _parse_locale_number(value: object) -> float:
    if pd.api.types.is_number(value):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return float("nan")
    text = text.replace("\xa0", "").replace(" ", "")
    text = text.replace(".", "").replace(",", ".")
    return float(text)


def _parse_imn_datetime(value: str) -> pd.Timestamp:
    text = unescape(str(value))
    text = text.replace("Â", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?)?", text)
    if not match:
        return pd.NaT
    day, month, year = map(int, match.group(1, 2, 3))
    hour = int(match.group(4) or 0)
    minute = int(match.group(5) or 0)
    second = int(match.group(6) or 0)
    am_pm = match.group(7)
    if am_pm == "a" and hour == 12:
        hour = 0
    elif am_pm == "p" and hour < 12:
        hour += 12
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


def _extract_tables(html: str) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for table_html in re.findall(r"<table.*?</table>", html, flags=re.S | re.I):
        headers = [_strip_tags(cell) for cell in re.findall(r"<th[^>]*>(.*?)</th>", table_html, flags=re.S | re.I)]
        rows = []
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I):
            cells = [_strip_tags(cell) for cell in re.findall(r"<td[^>]*>(.*?)</t[dh]>", row_html, flags=re.S | re.I)]
            if cells:
                rows.append(cells)
        if headers and rows:
            normalized_rows = [row[: len(headers)] for row in rows if len(row) >= len(headers)]
            tables.append(pd.DataFrame(normalized_rows, columns=headers))
    return tables


def normalize_weather_frame(df: pd.DataFrame, station: str | None = None) -> pd.DataFrame:
    rename = {
        "fecha": "timestamp",
        "timestamp": "timestamp",
        "date": "timestamp",
        "temp": "weather_temp_c",
        "temperatura": "weather_temp_c",
        "temperature": "weather_temp_c",
        "lluvia": "weather_precip_mm",
        "precip": "weather_precip_mm",
        "precipitation": "weather_precip_mm",
        "rad_max": "weather_rad_max",
        "radiacion": "weather_rad_max",
        "radiation": "weather_rad_max",
        "pres_atm": "weather_pressure_hpa",
        "p_atm": "weather_pressure_hpa",
        "hr": "weather_humidity_pct",
        "humedad": "weather_humidity_pct",
    }
    normalized_names = {column: rename.get(str(column).strip().lower(), str(column).strip()) for column in df.columns}
    weather = df.rename(columns=normalized_names).copy()
    if "timestamp" not in weather.columns:
        raise ValueError("weather data must include a Fecha/timestamp column")

    parsed_timestamp = weather["timestamp"].map(_parse_imn_datetime)
    missing_timestamp = parsed_timestamp.isna()
    if missing_timestamp.any():
        parsed_timestamp.loc[missing_timestamp] = pd.to_datetime(
            weather.loc[missing_timestamp, "timestamp"],
            errors="coerce",
        )
    weather["timestamp"] = parsed_timestamp
    for column in list(weather.columns):
        if column.startswith("weather_"):
            weather[column] = weather[column].map(_parse_locale_number)
    if station:
        weather["weather_station"] = station
    numeric_cols = ["timestamp"] + [column for column in weather.columns if column.startswith("weather_")]
    numeric_cols = [column for column in numeric_cols if column != "weather_station"]
    weather = weather[numeric_cols].dropna(subset=["timestamp"]).sort_values("timestamp")
    return weather.drop_duplicates("timestamp").reset_index(drop=True)


def read_weather_csv(path: str | Path) -> pd.DataFrame:
    return normalize_weather_frame(pd.read_csv(path))


def fetch_imn_station_hourly(station: str = "limon", timeout: int = 30) -> pd.DataFrame:
    station_key = station.lower()
    url = IMN_STATION_TABLES.get(station_key, station)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    tables = _extract_tables(response.text)
    if not tables:
        raise ValueError(f"No tables found for IMN station {station}")
    return normalize_weather_frame(tables[0], station=station_key)


def write_weather_csv(df: pd.DataFrame, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_weather_frame(df).to_csv(out_path, index=False)
    return out_path


def merge_weather_context(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame | None,
    frequency: str = "15min",
) -> pd.DataFrame:
    if weather_df is None or weather_df.empty:
        return demand_df.copy()

    demand = demand_df.copy()
    demand["timestamp"] = pd.to_datetime(demand["timestamp"]).astype("datetime64[ns]")
    weather = normalize_weather_frame(weather_df).set_index("timestamp").sort_index()
    weather_numeric = weather.select_dtypes(include="number")
    weather_regular = weather_numeric.resample(frequency).interpolate(method="time").ffill().bfill()
    weather_regular = weather_regular.reset_index()
    weather_regular["timestamp"] = pd.to_datetime(weather_regular["timestamp"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        demand.sort_values("timestamp"),
        weather_regular.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.to_timedelta(frequency),
    )
    return merged


def easter_date(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def costa_rica_event_days(year: int) -> dict[date, str]:
    easter = easter_date(year)
    events = {
        date(year, 1, 1): "Año Nuevo",
        easter - timedelta(days=3): "Jueves Santo",
        easter - timedelta(days=2): "Viernes Santo",
        date(year, 4, 11): "Batalla de Rivas",
        date(year, 5, 1): "Día del Trabajador",
        date(year, 7, 25): "Anexión del Partido de Nicoya",
        date(year, 8, 2): "Día de la Virgen de los Ángeles",
        date(year, 8, 15): "Día de la Madre",
        date(year, 9, 15): "Independencia",
        date(year, 12, 1): "Abolición del Ejército",
        date(year, 12, 24): "Nochebuena",
        date(year, 12, 25): "Navidad",
        date(year, 12, 31): "Fin de año",
    }
    return events


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    days = pd.Series(index.date, index=index)
    event_map: dict[date, str] = {}
    for year in range(index.min().year - 1, index.max().year + 2):
        event_map.update(costa_rica_event_days(year))

    event_dates = sorted(event_map)
    event_set = set(event_dates)
    features = pd.DataFrame(index=index)
    features["calendar_is_cr_holiday"] = days.isin(event_set).astype(int)
    features["calendar_is_christmas_season"] = (
        ((index.month == 12) & (index.day >= 1)) | ((index.month == 1) & (index.day <= 6))
    ).astype(int)
    features["calendar_is_school_vacation_estimate"] = (
        ((index.month == 12) & (index.day >= 15))
        | ((index.month == 1) & (index.day <= 15))
        | ((index.month == 7) & (index.day <= 14))
    ).astype(int)
    event_windows = set()
    for event_day in event_dates:
        event_windows.update({event_day - timedelta(days=1), event_day, event_day + timedelta(days=1)})
    features["event_is_window"] = days.isin(event_windows).astype(int)
    features["event_days_to_nearest"] = [
        min(abs((event_day - current_day).days) for event_day in event_dates) if event_dates else 999
        for current_day in days
    ]
    return features


def add_calendar_context(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    index = pd.DatetimeIndex(pd.to_datetime(enriched["timestamp"]))
    features = calendar_features(index).reset_index(drop=True)
    for column in features.columns:
        enriched[column] = features[column].to_numpy()
    return enriched


def add_enos_context(df: pd.DataFrame, phase: str | None) -> pd.DataFrame:
    enriched = df.copy()
    phase = (phase or "unknown").replace("-", "_").lower()
    enriched["enos_el_nino"] = int(phase == "el_nino")
    enriched["enos_la_nina"] = int(phase == "la_nina")
    enriched["enos_neutral"] = int(phase == "neutral")
    return enriched


def enrich_demand_context(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    enos_phase: str | None = None,
    frequency: str = "15min",
) -> pd.DataFrame:
    enriched = merge_weather_context(demand_df, weather_df, frequency=frequency)
    enriched = add_calendar_context(enriched)
    enriched = add_enos_context(enriched, enos_phase)
    return enriched
