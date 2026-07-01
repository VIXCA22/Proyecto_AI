from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def forecast_figure(cleaned: pd.DataFrame, forecast: pd.DataFrame, tail_points: int = 7 * 96) -> go.Figure:
    history = cleaned.tail(tail_points).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["mw_clean"],
            mode="lines",
            name="Historico limpio",
            line=dict(color="#2f6f9f", width=2),
        )
    )
    if "is_anomaly" in history:
        anomalies = history[history["is_anomaly"]]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies["timestamp"],
                    y=anomalies["mw_original"],
                    mode="markers",
                    name="Anomalias detectadas",
                    marker=dict(color="#d1495b", size=7),
                )
            )
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["prediction_mw"],
            mode="lines+markers",
            name="Prediccion",
            line=dict(color="#edae49", width=3),
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Fecha y hora",
        yaxis_title="Demanda MW",
        legend_title="Serie",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig
