# Agente para predicción de demanda eléctrica

Proyecto para IE0435: agente de análisis y predicción de series de tiempo para demanda eléctrica.

El proyecto toma datos de demanda del CENCE, limpia la serie, detecta y corrige anomalías, calcula un promedio horario histórico y genera una predicción de corto plazo.

## Alcance académico

Para el proyecto se seleccionaron dos herramientas:

- Detección y corrección de anomalías.
- Promedio horario histórico para la predicción.

## Estructura

```text
src/electroagent/
  datos.py         descarga y ordena datos CENCE
  cleaning.py      detecta anomalías y corrige valores
  features.py      genera variables de tiempo y promedio histórico
  models.py        aplica el promedio histórico y calcula métricas
  agente.py        orquesta el flujo completo
  cli.py           comandos de terminal
  visualization.py gráficos de resultados
app.py             panel Streamlit
tests/             pruebas automáticas
reportes/          informe del proyecto
```

## Instalación

Se recomienda usar Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Uso rápido

Descargar datos CENCE:

```powershell
electroagent download --start 2026-06-03 --end 2026-06-09 --out datos/raw/cence_junio_2026.csv
```

Ejecutar el agente:

```powershell
electroagent run --data datos/raw/cence_junio_2026.csv --horizon-steps 96
```

Abrir el panel:

```powershell
streamlit run app.py
```

## Flujo del agente

1. Recibe o descarga datos de demanda eléctrica.
2. Regulariza la serie a intervalos de 15 minutos.
3. Detecta valores faltantes o anómalos.
4. Corrige los puntos problemáticos mediante interpolación.
5. Calcula el promedio horario histórico.
6. Genera la predicción.
7. Compara la predicción contra los datos reales del CENCE en un tramo de validación.
8. Guarda resultados y gráficos.

## Resultado de demostración

Con datos CENCE del 2026-06-03 al 2026-06-16, la predicción con promedio horario histórico obtuvo:

| Métrica | Valor |
|---|---:|
| MAE | 95.60 MW |
| RMSE | 108.02 MW |
| MAPE | 5.99% |

En esa muestra no se detectaron anomalías, pero la herramienta de limpieza quedó implementada para datos faltantes o valores fuera del comportamiento esperado.

Además, se genera una gráfica de comparación entre la demanda real del CENCE y la predicción del agente. Esta comparación permite ver visualmente en qué momentos la predicción se acerca o se aleja de la curva real.

## Archivos de salida

- `outputs/cleaned.csv`: datos regularizados y limpios.
- `outputs/forecast.csv`: predicción generada.
- `outputs/metrics.csv`: métricas de validación.
- `outputs/forecast.html`: visualización de la predicción.
- `outputs/comparacion_cence_modelo.html`: comparación gráfica entre CENCE real y predicción.
- `reportes/comparacion_cence_modelo.html`: copia de la comparación junto al informe.
