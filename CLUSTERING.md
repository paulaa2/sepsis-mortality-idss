# Clustering de pacientes con sepsis

Este modulo implementa solo la parte de clustering del proyecto y deja la salida preparada para usarla despues con riesgo, XAI y base de conocimiento.

## Que compara

- `kmeans`
- `gmm`
- `birch`
- `agglomerative`
- `dbscan`

Por defecto se ejecutan `kmeans`, `gmm` y `birch`, que son los mas razonables para una primera pasada en esta base. `agglomerative` y `dbscan` siguen disponibles, pero conviene lanzarlos de forma explicita porque pueden tardar bastante mas.

## Entrada por defecto

- [bbdd/BaseDatos_imputada_knn.csv](C:/Users/Paess/Documents/GitHub/sepsis-mortality-idss/bbdd/BaseDatos_imputada_knn.csv)

El cargador detecta automaticamente si el CSV usa `;` con coma decimal o `,` con punto decimal.

## Ejecucion

```powershell
python .\src\icu_sepsis_clustering.py `
  --input ".\bbdd\BaseDatos_imputada_knn.csv" `
  --output-dir ".\outputs\clustering" `
  --min-k 3 `
  --max-k 8 `
  --verbose
```

## Ejecucion rapida para comprobar que todo corre

```powershell
python .\src\icu_sepsis_clustering.py `
  --input ".\bbdd\BaseDatos_imputada_knn.csv" `
  --output-dir ".\outputs\clustering_fast" `
  --methods kmeans birch `
  --min-k 3 `
  --max-k 5 `
  --max-rows 15000 `
  --verbose
```

## Comparar las dos estrategias que mas interesan para el trabajo

### 1. Enfoque basado en scores

Usa sobre todo variables agregadas de gravedad, suele dar clusters mas compactos y a veces mejor `silhouette`.

```powershell
python .\src\icu_sepsis_clustering.py `
  --input ".\bbdd\BaseDatos_imputada_knn.csv" `
  --output-dir ".\outputs\clustering_scores" `
  --feature-preset scores_only `
  --methods kmeans gmm birch `
  --min-k 3 `
  --max-k 5 `
  --verbose
```

### 2. Enfoque clinico fisiologico

Usa variables mas cercanas al estado del paciente y excluye columnas post-evento. Suele ser mas defendible clinicamente aunque no siempre maximice `silhouette`.

```powershell
python .\src\icu_sepsis_clustering.py `
  --input ".\bbdd\BaseDatos_imputada_knn.csv" `
  --output-dir ".\outputs\clustering_clinical" `
  --feature-preset clinical_core `
  --methods kmeans gmm birch `
  --min-k 3 `
  --max-k 5 `
  --verbose
```

### 3. Enfoque de senales crudas

Usa signos vitales y laboratorio en crudo, con menos dependencia de scores agregados.

```powershell
python .\src\icu_sepsis_clustering.py `
  --input ".\bbdd\BaseDatos_imputada_knn.csv" `
  --output-dir ".\outputs\clustering_raw" `
  --feature-preset raw_signals_only `
  --methods kmeans birch `
  --min-k 3 `
  --max-k 5 `
  --verbose
```

## Ejecucion ampliada con DBSCAN y jerarquico

```powershell
python .\src\icu_sepsis_clustering.py `
  --input ".\bbdd\BaseDatos_imputada_knn.csv" `
  --output-dir ".\outputs\clustering_full" `
  --methods kmeans gmm birch agglomerative dbscan `
  --dbscan-eps 1.0 1.2 `
  --dbscan-min-samples 50 `
  --verbose
```

## Si parece que no hace nada

- Usa siempre `--verbose` para ver el progreso.
- Si quieres validar primero que el pipeline funciona, usa `--max-rows 15000`.
- `dbscan` puede tardar bastante en datasets grandes.
- `gmm` tambien puede tardar bastante cuando hay muchas filas.
- `agglomerative` solo se ejecuta si el numero de filas no supera el limite configurado.
- `feature-preset all` excluye ya las columnas mas claras de leakage post-evento.

## Salidas

- `model_comparison.csv`: ranking de todos los experimentos
- `best_cluster_summary.csv`: resumen clinico por cluster
- `best_cluster_profiles.csv`: variables mas altas y mas bajas frente a la media global
- `best_patient_assignments.csv`: cluster asignado a cada paciente
- `run_metadata.json`: configuracion de la ejecucion

## Como encaja despues en el IDSS

- El clustering descubre fenotipos o perfiles de paciente.
- La prediccion supervisada de mortalidad puede anadirse despues sin cambiar este modulo.
- El resumen por cluster y los perfiles ayudan a enlazar los clusters con explicabilidad y reglas clinicas.
