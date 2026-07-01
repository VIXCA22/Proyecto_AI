from __future__ import annotations

import tempfile

import pandas as pd
import streamlit as st

from electroagent.agent import ForecastAgent
from electroagent.context import fetch_latest_enos_bulletin, read_weather_csv
from electroagent.data import fetch_cence_demand, read_demand_csv
from electroagent.visualization import forecast_figure


st.set_page_config(page_title="ElectroForecast Agent", layout="wide")

st.title("ElectroForecast Agent")
st.caption("Agente para analisis, limpieza y prediccion de demanda electrica.")

source = st.sidebar.radio("Fuente de datos", ["Descargar CENCE", "Subir CSV"])
horizon_hours = st.sidebar.slider("Horizonte (horas)", min_value=1, max_value=72, value=24)
horizon_steps = horizon_hours * 4
weather_upload = st.sidebar.file_uploader("CSV de clima IMN/exportado", type=["csv"])
use_latest_enos = st.sidebar.checkbox("Usar boletin ENOS del IMN", value=True)

weather_df = None
if weather_upload is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(weather_upload.read())
        weather_df = read_weather_csv(tmp.name)
        st.session_state["weather_df"] = weather_df
else:
    weather_df = st.session_state.get("weather_df")

enos_phase = "unknown"
if use_latest_enos:
    try:
        bulletin = fetch_latest_enos_bulletin()
        enos_phase = bulletin.phase
        st.sidebar.caption(f"ENOS: {bulletin.title} ({bulletin.phase})")
    except Exception as exc:
        st.sidebar.warning(f"No se pudo leer ENOS IMN: {exc}")

df = None
if source == "Descargar CENCE":
    start = st.sidebar.date_input("Fecha inicial")
    end = st.sidebar.date_input("Fecha final", value=start)
    if st.sidebar.button("Descargar datos"):
        with st.spinner("Descargando datos CENCE..."):
            df = fetch_cence_demand(start, end)
            st.session_state["df"] = df
else:
    uploaded = st.sidebar.file_uploader("CSV", type=["csv"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            df = read_demand_csv(tmp.name)
            st.session_state["df"] = df

df = st.session_state.get("df", df)

if df is None:
    st.info("Seleccione una fuente de datos para iniciar.")
    st.stop()

st.subheader("Datos cargados")
st.dataframe(df.tail(20), use_container_width=True)

if st.button("Ejecutar agente"):
    with st.spinner("Evaluando herramientas y generando prediccion..."):
        agent = ForecastAgent()
        result = agent.run(df, horizon_steps=horizon_steps, weather_df=weather_df, enos_phase=enos_phase)
        st.session_state["result"] = result

result = st.session_state.get("result")
if result is None:
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Modelo elegido", result.selection.name)
metric_cols[1].metric("Anomalias", result.cleaning_report.detected_anomalies)
metric_cols[2].metric("Puntos corregidos", result.cleaning_report.corrected_points)
metric_cols[3].metric("Demanda media MW", f"{result.summary['mean_mw']:.1f}")

if result.context_columns:
    st.caption("Variables de contexto: " + ", ".join(result.context_columns))

st.subheader("Prediccion")
st.plotly_chart(forecast_figure(result.cleaned, result.forecast), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Ranking de modelos")
    st.dataframe(result.selection.ranking, use_container_width=True)
with right:
    st.subheader("Reporte de limpieza")
    st.json(result.cleaning_report.__dict__)

st.subheader("Pronostico")
st.dataframe(result.forecast, use_container_width=True)
