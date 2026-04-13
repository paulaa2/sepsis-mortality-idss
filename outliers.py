"""
Analisis de outliers sobre la base de datos de sepsis imputada con KNN.

Objetivo:
1. Cargar la base imputada.
2. Seleccionar variables numericas relevantes.
3. Detectar outliers con IQR y z-score.
4. Guardar un resumen para revisar que variables necesitan limpieza.
5. Generar graficos para inspeccion visual.

Notas:
- Este script no elimina outliers automaticamente.
- Solo identifica variables y observaciones potencialmente anomalias.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


DATA_PATH = Path("bbdd") / "BaseDatos_imputada_knn.csv"
OUTPUT_DIR = Path("resultados_outliers")

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
    Carga la base de datos imputada.

    La base actual esta separada por ';'. Si en el futuro cambiara el
    formato, dejamos un fallback para intentar leerla como CSV estandar.
    """
    try:
        df = pd.read_csv(ruta_csv, sep=";")
    except Exception:
        df = pd.read_csv(ruta_csv)

    if len(df.columns) == 1:
        df = pd.read_csv(ruta_csv)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "H1" in df.columns:
        df = df.drop(columns=["H1"])

    return df


def obtener_columnas_numericas(df):
    """Devuelve solo las columnas numericas del DataFrame."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def convertir_columnas_a_numerico(df, columnas):
    """
    Convierte columnas objetivo a numerico de forma robusta.

    Si alguna columna llega como texto, intenta normalizarla y transforma
    valores no validos a NaN para no romper el analisis.
    """
    df_convertido = df.copy()

    for columna in columnas:
        if columna not in df_convertido.columns:
            continue

        serie = df_convertido[columna]

        if pd.api.types.is_numeric_dtype(serie):
            continue

        serie_limpia = (
            serie.astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
            .str.replace(",", ".", regex=False)
        )
        df_convertido[columna] = pd.to_numeric(serie_limpia, errors="coerce")

    return df_convertido


def seleccionar_variables_revision(df, variables_sugeridas=None):
    """
    Devuelve las variables que existen en el dataset.

    Si ninguna de las sugeridas aparece, usa todas las numericas.
    """
    if variables_sugeridas is None:
        variables_sugeridas = VARIABLES_PRIORITARIAS

    variables_existentes = [col for col in variables_sugeridas if col in df.columns]
    if variables_existentes:
        return variables_existentes

    return obtener_columnas_numericas(df)


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

    if not resumen:
        return pd.DataFrame()

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

    if not resumen:
        return pd.DataFrame()

    return pd.DataFrame(resumen).sort_values(by="n_outliers", ascending=False)


def guardar_resumenes(resumen_iqr, resumen_zscore, output_dir=OUTPUT_DIR):
    """Guarda los resumenes de outliers en CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not resumen_iqr.empty:
        resumen_iqr.to_csv(output_dir / "outliers_iqr_resumen.csv", index=False)

    if not resumen_zscore.empty:
        resumen_zscore.to_csv(output_dir / "outliers_zscore_resumen.csv", index=False)


def variables_top_outliers(resumen_iqr, limite=6):
    """Devuelve las variables con mas outliers segun IQR."""
    if resumen_iqr.empty:
        return []

    return resumen_iqr.head(limite)["variable"].tolist()


def mostrar_boxplots(df, columnas, max_cols=3, output_dir=OUTPUT_DIR):
    """Genera boxplots para revisar outliers visualmente."""
    if not columnas:
        print("No hay columnas para representar.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
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
    ruta_salida = output_dir / "boxplots_outliers.png"
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    print(f"Boxplots guardados en: {ruta_salida}")
    plt.show()
    plt.close(fig)


def mostrar_histogramas(df, columnas, max_cols=3, output_dir=OUTPUT_DIR):
    """Genera histogramas para ver la distribucion de cada variable."""
    if not columnas:
        print("No hay columnas para representar.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
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
    ruta_salida = output_dir / "histogramas_outliers.png"
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    print(f"Histogramas guardados en: {ruta_salida}")
    plt.show()
    plt.close(fig)


def mostrar_resumen_outliers(resumen_iqr, output_dir=OUTPUT_DIR):
    """Genera una grafica con el numero de outliers por variable."""
    if resumen_iqr.empty:
        print("No hay resumen IQR para representar.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    top_resumen = resumen_iqr.head(10).sort_values("n_outliers", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_resumen, x="n_outliers", y="variable", ax=ax, color="indianred")
    ax.set_title("Variables con mas outliers detectados por IQR")
    ax.set_xlabel("Numero de outliers")
    ax.set_ylabel("Variable")
    plt.tight_layout()

    ruta_salida = output_dir / "ranking_outliers_iqr.png"
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    print(f"Ranking de outliers guardado en: {ruta_salida}")
    plt.show()
    plt.close(fig)


def resumen_general(df):
    """Imprime un pequeno resumen de la base de datos."""
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
    df = convertir_columnas_a_numerico(df, variables_revision)
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

    # 6. Guardar resumenes para revision posterior
    guardar_resumenes(resumen_iqr, resumen_zscore)

    # 7. Representaciones graficas de las variables mas afectadas
    variables_graficar = variables_top_outliers(resumen_iqr, limite=6)
    if not variables_graficar:
        variables_graficar = variables_revision[:6]

    print("\n--- Variables representadas graficamente ---")
    print(variables_graficar)

    mostrar_resumen_outliers(resumen_iqr)
    mostrar_boxplots(df, variables_graficar)
    mostrar_histogramas(df, variables_graficar)


if __name__ == "__main__":
    main()
