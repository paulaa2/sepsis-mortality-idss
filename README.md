# Sepsis Mortality IDSS

Repositorio del sistema de soporte a la decision clinica para pacientes con sepsis en UCI.

## Estructura

- `Data/`: datos usados por los modelos.
- `Source/Training/`: scripts de entrenamiento de modelos.
- `Source/Analysis/`: EDA, outliers y profiling de clustering.
- `Source/Preprocessing/`: reservado para scripts de preprocesado activos.
- `IDSS/`: pipeline para generar el contexto de un nuevo paciente.
- `KnowledgeSources/`: guias clinicas y base de conocimiento.
- `Documentation/`: documentacion tecnica y explicativa del proyecto.
- `Web/`: interfaz web FastAPI y frontend del IDSS.
- `outputs/`: resultados generados por los modelos y analisis.
- `Deprecated/`: ficheros antiguos, legacy o no usados actualmente, conservados sin borrar.
- `SubmissionPackage/`: copia organizada para la entrega final.

## Scripts principales

- `Source/Training/icu_sepsis_clustering.py`: entrenamiento del clustering clinico.
- `Source/Training/XGBoost.py`: modelo supervisado de mortalidad y explicabilidad.
- `Source/Analysis/cluster_profiling.py`: profiling, TLP y aTLP de los clusters.
- `Source/Analysis/outliers.py`: analisis de outliers.
- `Source/LLM/ollama_explainer.py`: generacion de explicaciones con Ollama.
- `IDSS/new_patient_pipeline.py`: generacion del contexto final para un paciente nuevo.
- `Web/app.py`: backend web que conecta formulario, pipeline y LLM.

Los scripts principales calculan automaticamente la raiz del repositorio, por lo que pueden ejecutarse desde la raiz o indicando su ruta completa.
