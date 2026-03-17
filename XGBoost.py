"""
ESQUELETO DE GRADIENT BOOSTING PARA PREDICCIÓN DE MORTALIDAD POR SEPSIS

Contexto en el IDSS:
Este script asume que los datos ya han pasado por la etapa 1 (Imputación con KNN).
Recibe un DataFrame limpio y sin valores nulos.

Pasos que realiza este código:
1. Separar las variables clínicas (X) de la variable a predecir (y = mortalidad).
2. Dividir los datos en un conjunto de entrenamiento (para que el modelo aprenda) 
   y un conjunto de prueba (para validarlo).
3. Configurar e inicializar el modelo Gradient Boosting (XGBoost).
4. Entrenar el modelo con los datos de entrenamiento.
5. Generar predicciones probabilísticas (el % de riesgo de muerte) en los datos de prueba.
6. Evaluar el rendimiento inicial usando el Área Bajo la Curva (AUC-ROC), 
   que es la métrica estándar en medicina.
7. Retornar el modelo entrenado para que pase a la etapa 3 (Explicabilidad).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb # Asegúrate de instalarlo con: pip install xgboost

def entrenar_modelo_sepsis(df_limpio, columna_target):
    
    # 1. Separar variables independientes (X) de la variable objetivo (y)
    X = df_limpio.drop(columns=[columna_target])
    y = df_limpio[columna_target]
    
    # 2. División de datos (80% entrenamiento, 20% prueba)
    # random_state asegura que siempre se divida igual para poder replicar el experimento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Inicializar el modelo Gradient Boosting (XGBoost)
    # Aquí es donde más adelante podréis ajustar "hiperparámetros"
    gb_model = xgb.XGBClassifier(
        objective='binary:logistic', # Porque predecimos Vida (0) o Muerte (1)
        eval_metric='auc',           # Métrica de evaluación interna
        random_state=42,
        use_label_encoder=False
    )
    
    # 4. Entrenar el modelo
    print("Entrenando el modelo Gradient Boosting...")
    gb_model.fit(X_train, y_train)
    
    # 5. Hacer predicciones en el conjunto de prueba
    # Usamos predict_proba para obtener la PROBABILIDAD (ej. 0.85 de riesgo), no solo un 0 o 1
    probabilidades = gb_model.predict_proba(X_test)[:, 1] 
    predicciones_binarias = gb_model.predict(X_test)
    
    # 6. Evaluación del modelo
    auc = roc_auc_score(y_test, probabilidades)
    print(f"\n--- Resultados de la Validación ---")
    print(f"AUC-ROC Score: {auc:.4f} (1.0 es perfecto, 0.5 es aleatorio)")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, predicciones_binarias))
    
    # 7. Retornar el modelo (y los datos de prueba) para usarlos en el modelo de Explicabilidad
    return gb_model, X_train, X_test

# ==========================================
# CÓMO USAR ESTA FUNCIÓN (Ejemplo de ejecución)
# ==========================================
if __name__ == "__main__":
    # Suponiendo que 'datos_imputados_knn.csv' es la salida de vuestro primer modelo
    df_pacientes = pd.read_csv('datos_imputados_knn.csv')
    
    # EJEMPLO SIMULADO (borrar o comentar en el código real)
    modelo_entrenado, datos_train, datos_test = entrenar_modelo_sepsis(df_pacientes, 'hospital_expire_flag')
    pass