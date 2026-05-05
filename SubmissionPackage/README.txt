README.txt - Structure and contents of the ZIP file
===================================================

This ZIP-ready package contains the materials currently available for the sepsis mortality IDSS project, organized according to the requested delivery structure.

1. Documentation
----------------
Current contents:
- `Informe_IDSS_Sepsia_UCI.docx`
- `PW3-D1_GRUP3.pdf`
- `Gannt + Repartiment de tasques.xlsx`

This folder currently contains the available written documentation and planning material.

2. Data
-------
2.1 Raw
- `Supplementary Table 1. APSIII.csv`

2.2 Processed
- `BaseDatos_imputada_knn.csv`

2.3 QualityAnalysis
- `outliers_iqr_resumen.csv`
- `outliers_zscore_resumen.csv`
- `boxplots_outliers.png`
- `histogramas_outliers.png`
- `ranking_outliers_iqr.png`

This folder includes the raw table currently preserved in the project, the processed dataset used in the models, and the outlier analysis results.

3. KnowledgeSources
-------------------
Current contents:
- `knowledge_base.txt`
- `guia-actuacion-sepsis.pdf`
- `SEPSIS-DOCUMENTO-DE-CONSENSO.pdf`
- `surviving_sepsis_campaign__international.5.pdf`

This folder contains the clinical knowledge sources and textual knowledge base used to support the system.

4. Models
---------
4.1 ClusteringClinical
- `best_cluster_profiles.csv`
- `best_cluster_summary.csv`
- `best_patient_assignments.csv`
- `clustering_model.joblib`
- `clustering_preprocessor.joblib`
- `clustering_scaler.joblib`
- `clustering_svd.joblib`
- `model_comparison.csv`
- `reference_patient_embeddings.csv`
- `run_metadata.json`

4.1.1 Profiling
- `atlp_panel.png`
- `atlp_summary.csv`
- `atlp_uncertainty_heatmap.png`
- `categorical_association_tests.csv`
- `categorical_context_summary.csv`
- `cluster_profile_report.txt`
- `cluster_profile_table.csv`
- `numeric_association_tests.csv`
- `profiling_validation_report.md`

4.2 XGBoostMortality
- `cluster_explainability_summary.csv`
- `global_feature_importance.csv`
- `llm_ready_patient_context.csv`
- `metrics.json`
- `patient_explanations.csv`
- `preprocessor.joblib`
- `run_metadata.json`
- `test_predictions.csv`
- `xgboost_model.json`

This folder contains the interpretable outputs and serialized artifacts of the clustering and mortality prediction models.

5. Reasoning engines
--------------------
Current contents:
- Empty folder

No separate reasoning engine implementation file is available at this stage.

6. Source
---------
6.1 Training
- `icu_sepsis_clustering.py`
- `XGBoost.py`

6.2 Preprocessing
- `preprocessing_estandarizacion.py`
- `preprocessing_missings.py`

6.3 Analysis
- `cluster_profiling.py`
- `EDA_sepsis.ipynb`
- `outliers.py`

Root of Source:
- `requirements.txt`

This folder contains the source code used during development and training.

7. IDSS
-------
Current contents:
- `new_patient_pipeline.py`
- `new_patient.csv`

7.1 Prompts
- `new_patient_prompt.txt`

This folder contains the currently available files related to the final inference pipeline for new patients.

8. Demo
-------
Current contents:
- Empty folder

No demo video has been added yet.

9. Presentation
---------------


General note
------------
The package has been assembled from the current state of the project repository. Some requested deliverables, such as the final presentation and demo video, are still pending and therefore their folders remain empty.
