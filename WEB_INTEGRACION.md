# PAID - Pipeline Clinico con Web + XGBoost + Clustering + Ollama

Este proyecto monta un flujo completo para analizar un paciente a partir de:

- datos introducidos manualmente en una web,
- un CSV clinico subido por el usuario,
- un pipeline de prediccion y clustering,
- un generador de prompt clinico,
- y un LLM local ejecutado con Ollama.

El resultado final es un informe en lenguaje natural que vuelve a la web y se muestra en pantalla.

## Flujo general

El recorrido completo es este:

1. El usuario abre la web.
2. Introduce nombre, apellidos, edad, altura, peso, genero y etnia.
3. Sube un CSV con una unica fila del paciente.
4. La web envia todo al endpoint `POST /api/analizar`.
5. El backend mezcla los datos del formulario con la fila del CSV.
6. Se genera un CSV combinado temporal.
7. Ese CSV se pasa a `new_patient_pipeline.py`.
8. El pipeline calcula riesgo, cluster y genera un prompt clinico.
9. Ese prompt se pasa a `ollama_explainer.py`.
10. `ollama_explainer.py` llama a Ollama con el modelo `medgemma:4b`.
11. La respuesta del LLM se guarda en disco y se devuelve al frontend.
12. La web muestra el informe generado.

## Estructura relevante

- [Web/app.py](/C:/Users/veron/Documents/New%20project/PAID/Web/app.py): backend FastAPI y punto de integracion entre web y pipeline.
- [Web/index.html](/C:/Users/veron/Documents/New%20project/PAID/Web/index.html): interfaz web.
- [Web/script.js](/C:/Users/veron/Documents/New%20project/PAID/Web/script.js): logica cliente para enviar el formulario y pintar resultados.
- [new_patient_pipeline.py](/C:/Users/veron/Documents/New%20project/PAID/new_patient_pipeline.py): genera outputs estructurados y el prompt del nuevo paciente.
- [ollama_explainer.py](/C:/Users/veron/Documents/New%20project/PAID/ollama_explainer.py): llama al LLM local y guarda la explicacion.
- [new_patient.csv](/C:/Users/veron/Documents/New%20project/PAID/new_patient.csv): ejemplo de CSV esperado.
- [knowledge_base.txt](/C:/Users/veron/Documents/New%20project/PAID/knowledge_base.txt): base de conocimiento clinica usada por el prompt.

## Requisitos previos

Antes de arrancar, necesitas:

- Python 3.10 o superior.
- PowerShell en Windows.
- Ollama instalado localmente.
- El modelo `medgemma:4b` descargado en Ollama.

## Instalacion

Abre PowerShell y entra en la carpeta del proyecto:

```powershell
cd "C:\Users\veron\Documents\New project\PAID"
```

Crea un entorno virtual:

```powershell
python -m venv .venv
```

Activalo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activacion:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Actualiza `pip`:

```powershell
python -m pip install --upgrade pip
```

Instala dependencias del pipeline:

```powershell
python -m pip install -r requirements.txt
```

Instala dependencias de la web:

```powershell
python -m pip install -r Web\requirements.txt
```

## Dependencias que deben quedar instaladas

Entre otras, el proyecto necesita:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`
- `ollama`

## Artefactos necesarios

El pipeline no entrena modelos al vuelo para la web. Necesita artefactos ya generados.

Comprueba que existan estos archivos:

- `outputs\xgboost_explainability\preprocessor.joblib`
- `outputs\xgboost_explainability\xgboost_model.json`
- `outputs\xgboost_explainability\run_metadata.json`
- `outputs\xgboost_explainability\llm_ready_patient_context.csv`
- `outputs\clustering_clinical\clustering_preprocessor.joblib`
- `outputs\clustering_clinical\clustering_svd.joblib`
- `outputs\clustering_clinical\clustering_scaler.joblib`
- `outputs\clustering_clinical\reference_patient_embeddings.csv`
- `outputs\clustering_clinical\run_metadata.json`
- `outputs\clustering_clinical\profiling\cluster_profile_table.csv`
- `knowledge_base.txt`

Si falta alguno, el backend no podra completar el analisis.

## Preparar Ollama

Ollama tiene que estar instalado y escuchando en `http://localhost:11434`.

Arranca el servicio:

```powershell
ollama serve
```

En otra terminal, descarga el modelo si todavia no existe:

```powershell
ollama pull medgemma:4b
```

Puedes comprobar los modelos disponibles:

```powershell
ollama list
```

Puedes comprobar que el servicio responde:

```powershell
curl http://localhost:11434/api/tags
```

Si devuelve informacion de modelos, Ollama esta accesible.

## Probar primero el pipeline sin web

Antes de levantar la web, conviene validar el pipeline principal.

Con el entorno virtual activado:

```powershell
python new_patient_pipeline.py --patient-input new_patient.csv
```

Eso debe generar un prompt en:

- `outputs\gemini_prompts\new_patient_prompt.txt`

Despues prueba la llamada al LLM:

```powershell
python ollama_explainer.py --prompt-path outputs\gemini_prompts\new_patient_prompt.txt --model medgemma:4b
```

Eso debe crear una salida en:

- `outputs\ollama_responses\new_patient_explanation.txt`

Si estos dos pasos funcionan, la parte central del sistema esta correcta.

## Arrancar la web

Desde la raiz del proyecto, con el entorno virtual activado:

```powershell
python -m uvicorn Web.app:app --host 0.0.0.0 --port 8000 --reload
```

Abre despues en el navegador:

[http://localhost:8000](http://localhost:8000)

## Uso de la aplicacion

En la web:

1. Rellena los datos del paciente.
2. Sube un CSV.
3. Pulsa `Analizar Paciente`.

El frontend mostrara un estado de carga mientras:

- se genera el CSV combinado,
- se ejecuta el pipeline,
- se llama al LLM,
- y se recupera la respuesta final.

## Formato esperado del CSV

El archivo subido debe cumplir estas reglas:

- debe ser un archivo `.csv`,
- debe estar en UTF-8,
- debe contener exactamente una fila de datos,
- y lo ideal es que tenga las columnas clinicas usadas por el modelo.

Ejemplo base:

- `subject_id`
- `hadm_id`
- `stay_id`
- `gender`
- `admission_age`
- `ethnicity`
- `admission_type`
- `admission_location`
- `marital_status`
- `apsiii`
- `apsiii_prob`
- `hr_score`
- `mbp_score`
- `temp_score`
- `resp_rate_score`
- `hematocrit_score`
- `wbc_score`
- `creatinine_score`
- `uo_score`
- `bun_score`
- `sodium_score`
- `glucose_score`
- `gcs_score`
- `heart_rate_max`
- `heart_rate_min`
- `mbp_min`
- `mbp_max`
- `temperature_min`
- `temperature_max`
- `resp_rate_min`
- `resp_rate_max`
- `hematocrit_min`
- `hematocrit_max`
- `wbc_min`
- `wbc_max`
- `creatinine_min`
- `creatinine_max`
- `bun_min`
- `bun_max`
- `sodium_min`
- `sodium_max`
- `glucose_min`
- `glucose_max`
- `urineoutput`
- `gcs_unable`
- `gcs_eyes`
- `gcs_verbal`
- `gcs_motor`
- `sepsis3`

Puedes usar como referencia [new_patient.csv](/C:/Users/veron/Documents/New%20project/PAID/new_patient.csv).

## Que hace el backend exactamente

El endpoint principal esta en [Web/app.py](/C:/Users/veron/Documents/New%20project/PAID/Web/app.py).

Internamente hace esto:

1. Lee el CSV subido.
2. Detecta delimitador `,` o `;`.
3. Valida que haya exactamente una fila.
4. Mezcla esa fila con los datos del formulario.
5. Si faltan `admission_age`, `gender` o `ethnicity`, intenta rellenarlos desde el formulario.
6. Si faltan identificadores como `subject_id`, `hadm_id`, `stay_id`, `patient_id` o `row_id`, genera valores temporales.
7. Guarda el CSV combinado en `Web\runtime\inputs\`.
8. Ejecuta `new_patient_pipeline.py`.
9. Guarda el prompt en `outputs\gemini_prompts\`.
10. Ejecuta `ollama_explainer.py`.
11. Guarda la respuesta del LLM en `Web\runtime\llm_outputs\`.
12. Devuelve la respuesta al frontend.

## Archivos generados en ejecucion

Cuando se analiza un paciente, se generan estos artefactos:

- `Web\runtime\inputs\patient_<id>.csv`
- `outputs\gemini_prompts\patient_<id>_prompt.txt`
- `Web\runtime\llm_outputs\patient_<id>_response.txt`

## Arranque recomendado en dos terminales

Terminal 1:

```powershell
cd "C:\Users\veron\Documents\New project\PAID"
.\.venv\Scripts\Activate.ps1
ollama serve
```

Terminal 2:

```powershell
cd "C:\Users\veron\Documents\New project\PAID"
.\.venv\Scripts\Activate.ps1
python -m uvicorn Web.app:app --host 0.0.0.0 --port 8000 --reload
```

Luego abre:

[http://localhost:8000](http://localhost:8000)

## Problemas comunes

### `ModuleNotFoundError`

Faltan paquetes Python. Reinstala dependencias:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r Web\requirements.txt
```

### No conecta con Ollama

Comprueba:

- que `ollama serve` esta arrancado,
- que `ollama list` muestra `medgemma:4b`,
- que `curl http://localhost:11434/api/tags` responde.

### El CSV falla

Revisa:

- que el archivo esta en UTF-8,
- que tiene una sola fila,
- que tiene cabeceras correctas,
- que contiene las variables necesarias para el modelo.

### Faltan artefactos de modelos

Si faltan archivos dentro de `outputs\xgboost_explainability\` o `outputs\clustering_clinical\`, el pipeline fallara al cargar preprocessadores, embeddings o metadatos.

### El modelo de Ollama es otro

Ahora mismo el backend usa:

- `medgemma:4b`

Si quieres cambiarlo, modifica `DEFAULT_LLM_MODEL` en [Web/app.py](/C:/Users/veron/Documents/New%20project/PAID/Web/app.py:21).

## Comandos utiles de comprobacion

Ver paquetes instalados:

```powershell
python -m pip list
```

Comprobar version de Python:

```powershell
python --version
```

Comprobar modelos de Ollama:

```powershell
ollama list
```

Comprobar que la API de Ollama responde:

```powershell
curl http://localhost:11434/api/tags
```

Lanzar la web:

```powershell
python -m uvicorn Web.app:app --host 0.0.0.0 --port 8000 --reload
```

## Estado actual del proyecto

La integracion web-pipeline-LLM ya esta cableada.

Lo minimo que tiene que estar bien para que todo funcione es:

- dependencias Python instaladas,
- artefactos del modelo presentes,
- Ollama arrancado,
- `medgemma:4b` descargado,
- y un CSV de entrada compatible.

Si todo eso esta correcto, la aplicacion deberia analizar el paciente desde la web y mostrar el texto generado por el LLM en pantalla.
