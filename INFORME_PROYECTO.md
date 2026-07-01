# Reporte del Proyecto 2

## Agente para predicción de demanda eléctrica

## 1. Contexto del problema

La demanda eléctrica cambia durante el día. En la madrugada suele bajar, durante el día sube por el uso de comercios, casas e industrias, y en algunas horas puede cambiar más rápido.

Por eso, la demanda se puede estudiar como una serie de tiempo, porque cada dato tiene una fecha y una hora. En este proyecto se usaron datos del CENCE para construir un agente sencillo que limpia los datos y hace una predicción de corto plazo.

## 2. Qué se hizo

El proyecto se hizo con una idea simple: tomar datos reales de demanda eléctrica, revisarlos, corregir posibles problemas y generar una predicción usando el comportamiento histórico por hora.

El flujo principal fue:

1. Descargar o cargar datos del CENCE.
2. Ordenar los datos por fecha y hora.
3. Revisar si hay datos faltantes o valores extraños.
4. Corregir esos problemas cuando aparecen.
5. Calcular un promedio histórico por día y hora.
6. Generar la predicción.
7. Comparar la predicción con los datos reales del CENCE.

## 3. Herramientas escogidas

Para cumplir con la consigna se escogieron solo dos herramientas:

1. **Detección y corrección de anomalías.**  
   Esta parte revisa si hay datos vacíos, repetidos o valores que no se ven normales. Si aparece un problema, el agente lo corrige usando los datos cercanos.

2. **Promedio horario histórico para predecir.**  
   Esta parte usa el comportamiento normal de la demanda en horarios parecidos. Por ejemplo, compara lunes con lunes y horas parecidas con horas parecidas. Es una técnica sencilla, pero fácil de explicar.

Estas dos herramientas son importantes porque primero se cuida la calidad de los datos y luego se hace la predicción.

## 4. Resultados obtenidos

La prueba se hizo con datos del CENCE desde el 2026-06-03 hasta el 2026-06-16. Los datos estaban en intervalos de 15 minutos.

En esta corrida no se detectaron anomalías fuertes. Aun así, la limpieza quedó lista para funcionar si en otro archivo aparecen datos faltantes o valores raros.

La predicción obtuvo estos resultados:

| Métrica | Valor |
|---|---:|
| MAE | 95.60 MW |
| RMSE | 108.02 MW |
| MAPE | 5.99% |

La métrica más fácil de entender es el MAE. En este caso, el error promedio fue de aproximadamente 95.60 MW.

## 5. Comparación con CENCE

También se generó una gráfica para comparar los datos reales del CENCE con los datos calculados por el modelo. Esta comparación es importante porque permite ver si la predicción sigue la forma general de la demanda.

En la gráfica se observa que el modelo sigue la tendencia principal de la curva real: baja en horas de menor consumo y sube cuando la demanda aumenta. Sin embargo, no siempre llega exactamente al mismo valor, sobre todo en cambios rápidos o picos de consumo.

La comparación se guardó en:

- `outputs/comparacion_cence_modelo.html`
- `reportes/comparacion_cence_modelo.html`

## 6. Conclusión

El proyecto cumple con el objetivo principal: construir un agente sencillo que trabaje con datos reales del CENCE, limpie la serie y genere una predicción de demanda eléctrica.

Se escogieron dos herramientas de la consigna: detección y corrección de anomalías, y promedio horario histórico para predicción. La solución queda simple, entendible y suficiente para explicar cómo se pasó de los datos reales a una predicción evaluada con métricas básicas.

Como mejora futura, se podrían usar más meses de datos y agregar variables como temperatura o lluvia, pero para esta versión se dejó solamente lo esencial del proyecto.
