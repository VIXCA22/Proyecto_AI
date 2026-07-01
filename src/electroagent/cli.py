from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agent import ForecastAgent
from .context import (
    fetch_imn_station_hourly,
    fetch_latest_enos_bulletin,
    read_weather_csv,
    write_enos_bulletin,
    write_weather_csv,
)
from .data import fetch_cence_demand, read_demand_csv, write_demand_csv
from .visualization import forecast_figure


def cmd_download(args: argparse.Namespace) -> None:
    df = fetch_cence_demand(args.start, args.end, interval_minutes=args.interval)
    out_path = write_demand_csv(df, args.out)
    print(f"Downloaded {len(df)} rows to {out_path}")


def cmd_run(args: argparse.Namespace) -> None:
    df = read_demand_csv(args.data)
    weather_df = read_weather_csv(args.weather) if args.weather else None
    enos_phase = args.enos_phase
    if args.enos_latest:
        bulletin = fetch_latest_enos_bulletin()
        enos_phase = bulletin.phase
        print(f"Latest IMN ENOS bulletin: {bulletin.title} ({bulletin.phase})")
        if bulletin.description:
            print(bulletin.description)
    agent = ForecastAgent(validation_splits=args.splits, validation_test_size=args.test_size)
    result = agent.run(df, horizon_steps=args.horizon_steps, weather_df=weather_df, enos_phase=enos_phase)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = out_dir / "forecast.csv"
    metrics_path = out_dir / "metrics.csv"
    ranking_path = out_dir / "model_ranking.csv"
    cleaned_path = out_dir / "cleaned.csv"
    figure_path = out_dir / "forecast.html"

    result.forecast.to_csv(forecast_path, index=False)
    result.selection.metrics.to_csv(metrics_path, index=False)
    result.selection.ranking.to_csv(ranking_path, index=False)
    result.cleaned.to_csv(cleaned_path, index=False)
    forecast_figure(result.cleaned, result.forecast).write_html(figure_path)

    print(f"Best model: {result.selection.name}")
    print(result.selection.ranking.to_string(index=False))
    print(f"Forecast saved to {forecast_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Cleaned data saved to {cleaned_path}")
    print(f"Figure saved to {figure_path}")
    if result.context_columns:
        print(f"Context features used: {', '.join(result.context_columns)}")


def cmd_stats(args: argparse.Namespace) -> None:
    df = read_demand_csv(args.data)
    agent = ForecastAgent(validation_splits=2, validation_test_size=48)
    result = agent.run(df, horizon_steps=1)
    summary = pd.Series(result.summary)
    report = pd.Series(result.cleaning_report.__dict__)
    print("Demand summary")
    print(summary.to_string())
    print()
    print("Cleaning report")
    print(report.to_string())


def cmd_imn_station(args: argparse.Namespace) -> None:
    df = fetch_imn_station_hourly(args.station)
    out_path = write_weather_csv(df, args.out)
    print(f"Downloaded {len(df)} hourly IMN rows to {out_path}")


def cmd_enos(args: argparse.Namespace) -> None:
    bulletin = fetch_latest_enos_bulletin()
    print(f"{bulletin.title} | phase={bulletin.phase}")
    if bulletin.description:
        print(bulletin.description)
    print(bulletin.url)
    if args.out:
        out_path = write_enos_bulletin(bulletin, args.out)
        print(f"Saved ENOS bulletin metadata to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente de prediccion de demanda electrica.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Descarga datos CENCE a CSV.")
    download.add_argument("--start", required=True, help="Fecha inicial YYYY-MM-DD.")
    download.add_argument("--end", required=True, help="Fecha final YYYY-MM-DD.")
    download.add_argument("--interval", type=int, default=15, help="Intervalo CENCE en minutos.")
    download.add_argument("--out", default="data/raw/cence_demand.csv", help="CSV de salida.")
    download.set_defaults(func=cmd_download)

    run = subparsers.add_parser("run", help="Limpia, evalua modelos y predice.")
    run.add_argument("--data", required=True, help="CSV con columnas fechaHora/MW o timestamp/mw.")
    run.add_argument("--horizon-steps", type=int, default=96, help="Pasos a predecir; 96 = 24 h a 15 min.")
    run.add_argument("--splits", type=int, default=3, help="Folds de validacion temporal.")
    run.add_argument("--test-size", type=int, default=96, help="Tamano de cada fold de prueba.")
    run.add_argument("--weather", help="CSV de clima IMN/exportado con Fecha/Temp/Lluvia/Rad_max.")
    run.add_argument("--enos-phase", choices=["el_nino", "la_nina", "neutral", "unknown"], default="unknown")
    run.add_argument("--enos-latest", action="store_true", help="Lee el boletin ENOS mas reciente del IMN.")
    run.add_argument("--out-dir", default="outputs", help="Carpeta de resultados.")
    run.set_defaults(func=cmd_run)

    stats = subparsers.add_parser("stats", help="Genera estadisticas del dataset.")
    stats.add_argument("--data", required=True, help="CSV con demanda.")
    stats.set_defaults(func=cmd_stats)

    imn = subparsers.add_parser("imn-station", help="Descarga tabla horaria actual de una estacion IMN.")
    imn.add_argument("--station", default="limon", help="Alias limon/pavas o URL de tabla IMN.")
    imn.add_argument("--out", default="data/raw/imn_station_hourly.csv", help="CSV de salida.")
    imn.set_defaults(func=cmd_imn_station)

    enos = subparsers.add_parser("enos", help="Lee el boletin ENOS mas reciente del IMN.")
    enos.add_argument("--out", help="JSON de salida opcional.")
    enos.set_defaults(func=cmd_enos)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
