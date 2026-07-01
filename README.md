# ElectroForecast Agent

Proyecto para IE0435: agente de analisis y prediccion de series de tiempo para demanda electrica.

El proyecto toma datos de demanda del CENCE, limpia la serie, detecta y corrige anomalias, crea variables temporales, evalua varias herramientas de prediccion y selecciona automaticamente el modelo con menor error en validacion temporal.

## Alcance academico

Herramientas implementadas del enunciado:

- Deteccion y correccion de anomalias.
- Promedio historico por franja horaria como linea base.
- Prediccion con KNN.
- Prediccion con Support Vector Regression.
- Prediccion con arbol de regresion.
- Visualizacion de datos, predicciones y metricas.
- Estadisticas descriptivas de la demanda.
- Variables exogenas de clima, ENOS y calendario costarricense.

Aunque el enunciado permite escoger dos herramientas, aqui se implementan varias para que el agente pueda comparar alternativas y justificar su decision.

## Arquitectura

```text
src/electroagent/
  data.py          descarga y normaliza datos CENCE
  cleaning.py      regulariza timestamps, detecta anomalias y corrige valores
  features.py      genera variables de tiempo, rezagos y promedio historico
  context.py       clima IMN, boletin ENOS y calendario de eventos
  models.py        define modelos, validacion temporal y seleccion
  agent.py         orquesta el flujo completo
  cli.py           comandos de terminal
  visualization.py graficos Plotly
app.py             dashboard Streamlit
tests/             pruebas automaticas
reports/           reporte del proyecto
```

## Instalacion

Use Python 3.12 recomendado:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Uso rapido

Descargar datos CENCE:

```powershell
electroagent download --start 2026-06-03 --end 2026-06-09 --out data/raw/cence_junio_2026.csv
```

Ejecutar el agente:

```powershell
electroagent run --data data/raw/cence_junio_2026.csv --horizon-steps 96
```

Descargar contexto del IMN:

```powershell
electroagent enos --out data/raw/enos_latest.json
electroagent imn-station --station limon --out data/raw/imn_limon_hourly.csv
```

Ejecutar con clima y ENOS:

```powershell
electroagent run --data data/raw/cence_junio_2026.csv --weather data/raw/imn_limon_hourly.csv --enos-latest --horizon-steps 96 --out-dir outputs_context
```

Abrir dashboard:

```powershell
streamlit run app.py
```

## Como aprende el agente

El agente no escoge un modelo por intuicion. Sigue este proceso:

1. Limpia la serie y corrige puntos anomalos.
2. Construye variables predictivas: hora, dia de semana, rezagos, medias moviles, promedio historico, clima, ENOS y calendario.
3. Entrena cada herramienta con backtesting temporal recursivo.
4. Calcula MAE, RMSE y MAPE.
5. Selecciona el modelo con menor MAE promedio.
6. Reentrena ese modelo con todos los datos limpios y genera el pronostico.

Este flujo es defendible porque evita comparar modelos sobre datos futuros usados en entrenamiento. Ademas, si se pide un horizonte de 24 horas, el agente valida modelos pronosticando 24 horas completas, no solo el siguiente punto de 15 minutos.

## Clima, ENOS y calendario

El modulo `context.py` agrega tres tipos de variables exogenas:

- `weather_*`: temperatura, lluvia, radiacion y presion desde CSV del IMN o tablas horarias de estaciones automaticas.
- `enos_*`: fase climatica tomada del boletin ENOS del IMN.
- `calendar_*` y `event_*`: feriados y eventos relevantes de Costa Rica, como Semana Santa, independencia, Navidad y temporada de vacaciones.

La pagina de estaciones automaticas del IMN publica tablas de condiciones actuales por estacion. Esas tablas sirven para datos recientes; para entrenamiento historico de calidad se recomienda usar un CSV historico/exportado con columnas como `Fecha`, `Temp`, `Lluvia`, `Rad_max`, `Pres_atm` o equivalentes.

El boletin ENOS del IMN usado en la prueba fue `#197(junio)`, donde el sitio indica que El Nino afecta al pais. En el modelo se representa como `enos_el_nino = 1`.

## Resultado de demostracion

Con datos CENCE del 2026-06-03 al 2026-06-16 y un horizonte de 24 horas, el agente genero este ranking:

| Modelo | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| historical_average | 95.60 | 108.02 | 5.99% |
| svr | 143.53 | 154.21 | 9.19% |
| knn | 151.76 | 185.55 | 9.29% |
| decision_tree | 307.42 | 358.96 | 17.82% |

El modelo seleccionado fue `historical_average`. Esto no es una falla del agente: significa que, para ese rango de datos y ese horizonte, el patron historico por franja horaria fue mas estable que las alternativas de regresion.

## Referencia de datos

Los datos provienen del endpoint publico del CENCE:

```text
https://apps.grupoice.com/CenceWeb/data/sen/json/DemandaMW
```

El parametro de fecha debe enviarse como `YYYYMMDD`.

## Comandos utiles

```powershell
pytest
electroagent stats --data data/raw/cence_junio_2026.csv
electroagent run --data data/raw/cence_junio_2026.csv --horizon-steps 192 --out-dir outputs
electroagent run --data data/raw/cence_junio_2026.csv --weather data/raw/imn_limon_hourly.csv --enos-latest
```
