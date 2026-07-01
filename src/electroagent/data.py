from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


CENCE_DEMAND_URL = "https://apps.grupoice.com/CenceWeb/data/sen/json/DemandaMW"


def parse_date(value: str | date | datetime) -> date:
    """Parse a CLI/user date into a date object."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def cence_date(value: str | date | datetime) -> str:
    """CENCE expects dates as YYYYMMDD, not ISO with separators."""
    return parse_date(value).strftime("%Y%m%d")


def iter_days(start: str | date | datetime, end: str | date | datetime) -> Iterable[date]:
    current = parse_date(start)
    last = parse_date(end)
    if current > last:
        raise ValueError("start date must be before or equal to end date")
    while current <= last:
        yield current
        current += timedelta(days=1)


def normalize_demand_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical demand dataframe with timestamp, mw and mw_programmed."""
    rename_map = {
        "fechaHora": "timestamp",
        "fecha": "timestamp",
        "MW": "mw",
        "demanda": "mw",
        "MW_P": "mw_programmed",
    }
    normalized = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    required = {"timestamp", "mw"}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    normalized["mw"] = pd.to_numeric(normalized["mw"], errors="coerce")
    if "mw_programmed" in normalized.columns:
        normalized["mw_programmed"] = pd.to_numeric(normalized["mw_programmed"], errors="coerce")
    else:
        normalized["mw_programmed"] = pd.NA

    normalized = normalized[["timestamp", "mw", "mw_programmed"]]
    normalized = normalized.sort_values("timestamp").drop_duplicates("timestamp")
    return normalized.reset_index(drop=True)


def fetch_cence_demand(
    start: str | date | datetime,
    end: str | date | datetime,
    interval_minutes: int = 15,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download demand data from CENCE one day at a time.

    The one-day loop is intentional: it makes retries and missing-day diagnosis
    easier, and it avoids large responses during demonstrations.
    """
    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for day in iter_days(start, end):
            params = {
                "intervalo": interval_minutes,
                "inicio": cence_date(day),
                "fin": cence_date(day),
            }
            response = session.get(CENCE_DEMAND_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            records = payload.get("data", [])
            if records:
                frames.append(pd.DataFrame(records))

    if not frames:
        raise ValueError("CENCE did not return demand records for the requested range")

    return normalize_demand_frame(pd.concat(frames, ignore_index=True))


def read_demand_csv(path: str | Path) -> pd.DataFrame:
    return normalize_demand_frame(pd.read_csv(path))


def write_demand_csv(df: pd.DataFrame, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_demand_frame(df).to_csv(out_path, index=False)
    return out_path
