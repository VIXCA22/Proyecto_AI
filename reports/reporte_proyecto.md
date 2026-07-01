# Reporte del Proyecto 2

## Contextualizacion

La demanda electrica es una serie de tiempo con patrones diarios, semanales y estacionales. En operacion del sistema electrico, una prediccion de las siguientes horas ayuda a planificar generacion, reservas y seguimiento operativo. El CENCE publica datos de demanda real y programada del Sistema Electrico Nacional, lo cual permite construir un caso realista para el curso.

## Objetivo

Desarrollar un agente capaz de seleccionar una herramienta de prediccion de demanda electrica a partir de datos disponibles, usando criterios cuantitativos de error y un flujo reproducible.

## Herramientas desarrolladas

- Deteccion y correccion de anomalias con mediana movil y MAD.
- Promedio historico por dia de semana y franja horaria.
- Regresion KNN.
- Support Vector Regression lineal.
- Arbol de decision para regresion.
- Estadisticas descriptivas y visualizacion interactiva.
- Variables de clima desde IMN, contexto ENOS y calendario de eventos de Costa Rica.

## Metodologia

1. Descarga de datos desde el endpoint publico del CENCE.
2. Normalizacion de columnas a `timestamp`, `mw` y `mw_programmed`.
3. Regularizacion de la serie a intervalos de 15 minutos.
4. Deteccion de valores atipicos mediante desviacion robusta respecto a la mediana movil.
5. Correccion de valores faltantes o anomalos con interpolacion temporal.
6. Generacion de variables: hora, dia de semana, dia del anio, rezagos, medias moviles, promedio historico, clima, ENOS y eventos.
7. Evaluacion de modelos con backtesting temporal recursivo.
8. Seleccion del modelo con menor MAE promedio.
9. Pronostico recursivo del horizonte solicitado.

## Agente

El agente no depende de un LLM para tomar la decision principal. Su comportamiento se define como una politica de seleccion de herramientas basada en metricas. Esto reduce respuestas no deterministas y hace que el resultado sea explicable:

- Si un modelo tiene menor MAE en validacion temporal, se selecciona.
- Si los datos contienen anomalias, se corrigen antes de entrenar.
- Si una herramienta no supera el promedio historico, el agente conserva la linea base.
- Si hay variables exogenas disponibles y alineadas temporalmente, las incorpora al entrenamiento; si no hay solape temporal, no fuerza esas variables.

## Incorporacion de clima y eventos

Se agrego una capa de contexto para capturar factores externos que pueden modificar la demanda electrica:

- Temperatura: dias mas calidos pueden aumentar el uso de aire acondicionado.
- Lluvia y radiacion: afectan comportamiento de ocupacion, iluminacion y confort termico.
- ENOS: el boletin ENOS del IMN permite codificar condiciones regionales como El Nino, La Nina o neutral.
- Calendario: feriados, Semana Santa, Navidad, vacaciones y fechas de celebracion pueden cambiar patrones residenciales, comerciales e industriales.

El sistema puede descargar condiciones horarias recientes de estaciones automaticas del IMN mediante:

```powershell
electroagent imn-station --station limon --out data/raw/imn_limon_hourly.csv
```

Tambien puede leer el boletin ENOS mas reciente:

```powershell
electroagent enos --out data/raw/enos_latest.json
```

Para entrenamiento historico completo, lo ideal es usar un archivo meteorologico historico/exportado del IMN con columnas como `Fecha`, `Temp`, `Lluvia`, `Rad_max`, `Pres_atm`. El agente lo alinea con la demanda mediante timestamp y solo usa esas variables cuando existen datos solapados.

## Resultados de una corrida de demostracion

Se descargaron datos CENCE desde el 2026-06-03 hasta el 2026-06-16. La serie contiene 1344 registros con frecuencia de 15 minutos. No se detectaron timestamps faltantes ni anomalias en esta corrida.

Estadisticas principales:

| Indicador | Valor |
|---|---:|
| Demanda media | 1586.94 MW |
| Mediana | 1635.99 MW |
| Desviacion estandar | 262.18 MW |
| Minimo | 1071.44 MW |
| Maximo | 2005.34 MW |

Ranking del backtesting para un horizonte de 24 horas:

| Modelo | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| historical_average | 95.60 | 108.02 | 5.99% |
| svr | 143.53 | 154.21 | 9.19% |
| knn | 151.76 | 185.55 | 9.29% |
| decision_tree | 307.42 | 358.96 | 17.82% |

El agente selecciono `historical_average`. Este resultado muestra por que es importante conservar una linea base robusta: con una ventana corta de entrenamiento, el promedio historico por dia de semana y franja horaria puede generalizar mejor que modelos mas flexibles.

Archivos generados:

- `outputs/forecast.csv`: prediccion de 24 horas.
- `outputs/model_ranking.csv`: ranking promedio de modelos.
- `outputs/metrics.csv`: metricas por fold.
- `outputs/cleaned.csv`: datos regularizados y limpios.
- `outputs/forecast.html`: visualizacion interactiva.

Con contexto IMN/ENOS se genero una segunda salida en `outputs_context/`. En esta prueba el boletin ENOS activo fue `#197(junio)`, clasificado como `el_nino`. El clima horario descargado desde la estacion de Limon corresponde a datos recientes; al no cubrir el mismo periodo de la muestra historica de demanda, el entrenamiento uso ENOS y calendario, pero no forzo clima sin solape temporal.

## Oportunidades de desarrollo futuro

- Agregar variables meteorologicas de acceso libre.
- Incluir feriados nacionales y calendario escolar/laboral.
- Implementar Prophet, SARIMA o modelos de gradiente boosting.
- Crear intervalos de prediccion para cuantificar incertidumbre.
- Integrar un LLM local como interfaz conversacional encima de este agente determinista.
- Automatizar ejecucion diaria y publicacion de reportes.
