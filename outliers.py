"""
ESQUELETO PARA ANALISIS DE OUTLIERS EN LA BASE DE DATOS DE SEPSIS

Objetivo:
1. Cargar la base de datos principal.
2. Separar columnas numericas utiles para el analisis.
3. Detectar posibles outliers con metodos sencillos.
4. Generar un resumen para revisar que variables necesitan limpieza.
5. Dibujar graficos basicos para inspeccion visual.

Notas:
- Este script no elimina outliers automaticamente.
- Primero los identifica y resume para que el equipo decida si:
  a) mantenerlos,
  b) caparlos,
  c) imputarlos,
  d) eliminarlos.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Ruta por defecto a la base de datos principal del proyecto
DATA_PATH = Path("bbdd") / "Supplementary Table 3. SEPSIS_FINAL.csv"

# Columnas que normalmente interesa revisar con prioridad clinica
VARIABLES_PRIORITARIAS = [
    "admission_age",
    "apsiii",
    "apsiii_prob",
    "heart_rate_max",
    "heart_rate_min",
    "mbp_max",
    "mbp_min",
    "temperature_max",
    "temperature_min",
    "resp_rate_max",
    "resp_rate_min",
    "wbc_max",
    "wbc_min",
    "creatinine_max",
    "creatinine_min",
    "bun_max",
    "bun_min",
    "sodium_max",
    "sodium_min",
    "glucose_max",
    "glucose_min",
    "urineoutput",
]


def cargar_datos(ruta_csv=DATA_PATH):
    """
    Carga la base de datos desde CSV.

    El dataset actual trae una primera columna tipo indice sin nombre,
    por eso se renombra y se elimina si aparece.
    """
    df = pd.read_csv(ruta_csv)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "H1" in df.columns:
        df = df.drop(columns=["H1"])

    return df


def obtener_columnas_numericas(df):
    """Devuelve solo las columnas numericas del DataFrame."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def seleccionar_variables_revision(df, variables_sugeridas=None):
    """
    Devuelve las variables que realmente existen en el dataset.
    Si no se pasa una lista, usa las variables clinicas prioritarias.
    """
    if variables_sugeridas is None:
        variables_sugeridas = VARIABLES_PRIORITARIAS

    return [col for col in variables_sugeridas if col in df.columns]


def detectar_outliers_iqr(df, columnas):
    """
    Detecta outliers por regla de IQR.

    Marca valores fuera de:
    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    """
    resumen = []

    for columna in columnas:
        serie = df[columna].dropna()
        if serie.empty:
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        mascara = (df[columna] < limite_inferior) | (df[columna] > limite_superior)
        n_outliers = int(mascara.sum())
        porcentaje = (n_outliers / len(df)) * 100 if len(df) else 0

        resumen.append(
            {
                "variable": columna,
                "metodo": "IQR",
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": limite_inferior,
                "limite_superior": limite_superior,
                "n_outliers": n_outliers,
                "porcentaje_outliers": porcentaje,
            }
        )

    return pd.DataFrame(resumen).sort_values(by="n_outliers", ascending=False)


def detectar_outliers_zscore(df, columnas, umbral=3.0):
    """
    Detecta outliers usando z-score.

    Un valor se considera outlier si:
    abs((x - media) / desviacion_std) > umbral
    """
    resumen = []

    for columna in columnas:
        serie = df[columna].dropna()
        if serie.empty or serie.std() == 0:
            continue

        media = serie.mean()
        desviacion = serie.std()
        z_scores = ((df[columna] - media) / desviacion).abs()
        mascara = z_scores > umbral

        n_outliers = int(mascara.sum())
        porcentaje = (n_outliers / len(df)) * 100 if len(df) else 0

        resumen.append(
            {
                "variable": columna,
                "metodo": "zscore",
                "media": media,
                "desviacion_std": desviacion,
                "umbral": umbral,
                "n_outliers": n_outliers,
                "porcentaje_outliers": porcentaje,
            }
        )

    return pd.DataFrame(resumen).sort_values(by="n_outliers", ascending=False)


def mostrar_boxplots(df, columnas, max_cols=3):
    """Genera boxplots para revisar outliers visualmente."""
    if not columnas:
        print("No hay columnas para representar.")
        return

    n_cols = min(max_cols, len(columnas))
    n_rows = (len(columnas) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = pd.Series(axes.ravel() if hasattr(axes, "ravel") else [axes])

    for ax, columna in zip(axes, columnas):
        sns.boxplot(y=df[columna], ax=ax, color="salmon")
        ax.set_title(f"Boxplot - {columna}")
        ax.set_ylabel(columna)

    for ax in axes[len(columnas):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def mostrar_histogramas(df, columnas, max_cols=3):
    """Genera histogramas para ver la distribucion de cada variable."""
    if not columnas:
        print("No hay columnas para representar.")
        return

    n_cols = min(max_cols, len(columnas))
    n_rows = (len(columnas) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = pd.Series(axes.ravel() if hasattr(axes, "ravel") else [axes])

    for ax, columna in zip(axes, columnas):
        sns.histplot(df[columna].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Histograma - {columna}")

    for ax in axes[len(columnas):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def resumen_general(df):
    """Imprime un pequeño resumen de la base de datos."""
    print("\n--- Resumen de la base de datos ---")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print("\nTipos de variables:")
    print(df.dtypes.value_counts())
    print("\nValores nulos por columna:")
    print(df.isnull().sum().sort_values(ascending=False).head(15))


def main():
    # 1. Cargar base de datos
    df = cargar_datos()

    # 2. Resumen inicial
    resumen_general(df)

    # 3. Elegir variables a revisar
    variables_revision = seleccionar_variables_revision(df)
    print("\n--- Variables seleccionadas para revisar outliers ---")
    print(variables_revision)

    # 4. Detectar outliers con IQR
    resumen_iqr = detectar_outliers_iqr(df, variables_revision)
    print("\n--- Resumen de outliers por IQR ---")
    print(resumen_iqr.head(15))

    # 5. Detectar outliers con z-score
    resumen_zscore = detectar_outliers_zscore(df, variables_revision, umbral=3.0)
    print("\n--- Resumen de outliers por z-score ---")
    print(resumen_zscore.head(15))

    # 6. Visualizacion basica
    mostrar_boxplots(df, variables_revision[:6])
    mostrar_histogramas(df, variables_revision[:6])

    # 7. Si quieres guardar resultados:
    # resumen_iqr.to_csv("outliers_iqr_resumen.csv", index=False)
    # resumen_zscore.to_csv("outliers_zscore_resumen.csv", index=False)


if __name__ == "__main__":
    main()
