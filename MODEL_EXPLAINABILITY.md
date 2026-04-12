# Modelo y explicabilidad

Este modulo entrena un `XGBoost` para predecir mortalidad hospitalaria por sepsis y genera salidas de explicabilidad reutilizables para el IDSS.

## Idea recomendada para el proyecto

Se mantiene el `clustering_clinical` como una capa separada de contexto:

- El `XGBoost` aprende el riesgo de mortalidad.
- La explicabilidad del modelo explica por que sube o baja ese riesgo.
- El `clustering_clinical` se mezcla despues en las salidas solo como contexto interpretativo.

Esto es una buena decision para el trabajo porque evita mezclar el clustering dentro del modelo supervisado y deja una salida mas limpia para el LLM y la knowledge base:

- riesgo predicho
- factores explicativos principales
- fenotipo clinico
- recomendacion basada en reglas o LLM

## Ejecucion basica

```powershell
python .\XGBoost.py `
  --input ".\bbdd\BaseDatos_imputada_knn.csv" `
  --output-dir ".\outputs\xgboost_explainability"
```

## Ejecucion con progreso y muestra rapida

```powershell
python .\XGBoost.py `
  --input ".\bbdd\BaseDatos_imputada_knn.csv" `
  --output-dir ".\outputs\xgboost_explainability_fast" `
  --max-rows 15000
```

## Que columnas se excluyen del modelo

Por defecto se excluyen como features:

- `deathoffset`
- `unitdischargeoffset`
- `hospitaldischargeoffset`
- `los_icu`
- `los_hospital`

Estas columnas se consideran problematicas para un modelo clinico porque estan muy ligadas al desenlace o a informacion posterior.

## Salidas

- `metrics.json`: metricas del modelo
- `global_feature_importance.csv`: importancia global por contribucion media absoluta
- `test_predictions.csv`: predicciones en test con grupo de riesgo
- `patient_explanations.csv`: explicacion local por paciente
- `llm_ready_patient_context.csv`: salida por paciente lista para pasar al LLM
- `cluster_explainability_summary.csv`: resumen de explicaciones agregadas por cluster
- `xgboost_model.json`: modelo entrenado
- `preprocessor.joblib`: transformador para volver a usar el modelo
- `run_metadata.json`: parametros usados

## Como leer las explicaciones

- `predicted_probability`: probabilidad de mortalidad predicha por el modelo
- `predicted_risk_group`: `low`, `medium` o `high`
- `top_positive_features`: variables que empujan el riesgo hacia arriba
- `top_negative_features`: variables que empujan el riesgo hacia abajo

## Uso con el clustering

Si existe `outputs/clustering_clinical/best_patient_assignments.csv`, el script lo mezcla automaticamente en las salidas, pero solo como contexto.

Eso permite construir despues una entrada al LLM del tipo:

- riesgo predicho alto
- fenotipo clinico `cluster_2`
- factores principales: `apsiii`, `creatinine_score`, `uo_score`
- recomendacion desde la knowledge base

La salida `llm_ready_patient_context.csv` ya deja esa informacion empaquetada en un campo `llm_summary`.
