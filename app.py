from __future__ import annotations

import tempfile

import streamlit as st

from electroagent.agente import AgentePrediccion
from electroagent.datos import fetch_cence_demand, read_demand_csv
from electroagent.visualization import forecast_figure, validation_comparison_figure


st.set_page_config(page_title="Agente de predicción eléctrica", layout="wide")

st.title("Agente de predicción eléctrica")
st.caption("Agente para análisis, limpieza y predicción de demanda eléctrica.")

source = st.sidebar.radio("Fuente de datos", ["Descargar CENCE", "Subir CSV"])
horizon_hours = st.sidebar.slider("Horizonte (horas)", min_value=1, max_value=72, value=24)
horizon_steps = horizon_hours * 4

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
    with st.spinner("Limpiando datos y generando predicción..."):
        agent = AgentePrediccion()
        result = agent.run(df, horizon_steps=horizon_steps)
        st.session_state["result"] = result

result = st.session_state.get("result")
if result is None:
    st.stop()

method_labels = {"promedio_historico": "Promedio histórico"}

metric_cols = st.columns(4)
metric_cols[0].metric("Método", method_labels.get(result.selection.name, result.selection.name))
metric_cols[1].metric("Anomalías", result.cleaning_report.detected_anomalies)
metric_cols[2].metric("Puntos corregidos", result.cleaning_report.corrected_points)
metric_cols[3].metric("Demanda media MW", f"{result.summary['mean_mw']:.1f}")

st.subheader("Comparación con datos reales del CENCE")
st.caption("Curva real publicada por CENCE frente a la predicción del agente en el tramo de validación.")
st.plotly_chart(validation_comparison_figure(result.validation_comparison), use_container_width=True)

st.subheader("Predicción")
st.plotly_chart(forecast_figure(result.cleaned, result.forecast), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Métricas de validación")
    ranking = result.selection.ranking.copy()
    if "model" in ranking:
        ranking["model"] = ranking["model"].replace(method_labels)
    ranking = ranking.rename(columns={"model": "método", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE"})
    st.dataframe(ranking, use_container_width=True)
with right:
    st.subheader("Reporte de limpieza")
    st.json(result.cleaning_report.__dict__)

st.subheader("Pronóstico")
st.dataframe(result.forecast, use_container_width=True)
