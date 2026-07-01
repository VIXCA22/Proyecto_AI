from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agente import AgentePrediccion
from .datos import fetch_cence_demand, read_demand_csv, write_demand_csv
from .visualization import forecast_figure, validation_comparison_figure


def cmd_download(args: argparse.Namespace) -> None:
    df = fetch_cence_demand(args.start, args.end, interval_minutes=args.interval)
    out_path = write_demand_csv(df, args.out)
    print(f"Se descargaron {len(df)} filas en {out_path}")


def cmd_run(args: argparse.Namespace) -> None:
    df = read_demand_csv(args.data)
    agent = AgentePrediccion(validation_splits=args.splits, validation_test_size=args.test_size)
    result = agent.run(df, horizon_steps=args.horizon_steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = out_dir / "forecast.csv"
    metrics_path = out_dir / "metrics.csv"
    ranking_path = out_dir / "model_ranking.csv"
    cleaned_path = out_dir / "cleaned.csv"
    figure_path = out_dir / "forecast.html"
    comparison_path = out_dir / "comparacion_cence_modelo.csv"
    comparison_figure_path = out_dir / "comparacion_cence_modelo.pdf"

    result.forecast.to_csv(forecast_path, index=False)
    result.selection.metrics.to_csv(metrics_path, index=False)
    result.selection.ranking.to_csv(ranking_path, index=False)
    result.cleaned.to_csv(cleaned_path, index=False)
    forecast_figure(result.cleaned, result.forecast).write_html(figure_path)
    result.validation_comparison.to_csv(comparison_path, index=False)
    validation_comparison_figure(result.validation_comparison).write_image(comparison_figure_path)

    print(f"Método de predicción: {result.selection.name}")
    print(result.selection.ranking.rename(columns={"model": "metodo"}).to_string(index=False))
    print(f"Predicción guardada en {forecast_path}")
    print(f"Métricas guardadas en {metrics_path}")
    print(f"Datos limpios guardados en {cleaned_path}")
    print(f"Gráfico guardado en {figure_path}")
    print(f"Comparación CENCE-modelo guardada en {comparison_figure_path}")


def cmd_stats(args: argparse.Namespace) -> None:
    df = read_demand_csv(args.data)
    agent = AgentePrediccion(validation_splits=2, validation_test_size=48)
    result = agent.run(df, horizon_steps=1)
    summary = pd.Series(result.summary)
    report = pd.Series(result.cleaning_report.__dict__)
    print("Resumen de demanda")
    print(summary.to_string())
    print()
    print("Reporte de limpieza")
    print(report.to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente de predicción de demanda eléctrica.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Descarga datos CENCE a CSV.")
    download.add_argument("--start", required=True, help="Fecha inicial YYYY-MM-DD.")
    download.add_argument("--end", required=True, help="Fecha final YYYY-MM-DD.")
    download.add_argument("--interval", type=int, default=15, help="Intervalo CENCE en minutos.")
    download.add_argument("--out", default="datos/raw/cence_demand.csv", help="CSV de salida.")
    download.set_defaults(func=cmd_download)

    run = subparsers.add_parser("run", help="Limpia datos y predice con promedio histórico.")
    run.add_argument("--data", required=True, help="CSV con columnas fechaHora/MW o timestamp/mw.")
    run.add_argument("--horizon-steps", type=int, default=96, help="Pasos a predecir; 96 = 24 h a 15 min.")
    run.add_argument("--splits", type=int, default=3, help="Pliegues de validación temporal.")
    run.add_argument("--test-size", type=int, default=96, help="Tamaño de cada tramo de prueba.")
    run.add_argument("--out-dir", default="outputs", help="Carpeta de resultados.")
    run.set_defaults(func=cmd_run)

    stats = subparsers.add_parser("stats", help="Genera estadísticas del conjunto de datos.")
    stats.add_argument("--data", required=True, help="CSV con demanda.")
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
