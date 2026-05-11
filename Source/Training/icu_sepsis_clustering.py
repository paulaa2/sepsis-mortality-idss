from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    return current.parent


REPO_ROOT = find_repo_root()
DEFAULT_INPUT = REPO_ROOT / "Data" / "Processed" / "BaseDatos_imputada_knn.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "clustering_clinical"
DEFAULT_CATEGORICAL_COLUMNS = [
    "gender",
    "ethnicity",
    "admission_type",
    "admission_location",
    "marital_status",
    "gcs_unable",
]
DEFAULT_ID_COLUMNS = [
    "H1",
    "Unnamed: 0",
    "subject_id",
    "hadm_id",
    "stay_id",
    "patient_id",
]
DEFAULT_METHODS = ["kmeans", "gmm", "birch"]
DEFAULT_DBSCAN_EPS = [1.0, 1.2]
DEFAULT_DBSCAN_MIN_SAMPLES = [50]
POST_OUTCOME_COLUMNS = [
    "deathoffset",
    "unitdischargeoffset",
    "hospitaldischargeoffset",
    "los_icu",
    "los_hospital",
]
SCORE_FAMILY_COLUMNS = [
    "apsiii",
    "apsiii_prob",
    "hr_score",
    "mbp_score",
    "temp_score",
    "resp_rate_score",
    "hematocrit_score",
    "wbc_score",
    "creatinine_score",
    "uo_score",
    "bun_score",
    "sodium_score",
    "glucose_score",
    "gcs_score",
]
RAW_SIGNAL_COLUMNS = [
    "heart_rate_max",
    "heart_rate_min",
    "mbp_min",
    "mbp_max",
    "temperature_min",
    "temperature_max",
    "resp_rate_min",
    "resp_rate_max",
    "hematocrit_min",
    "hematocrit_max",
    "wbc_min",
    "wbc_max",
    "creatinine_min",
    "creatinine_max",
    "bun_min",
    "bun_max",
    "sodium_min",
    "sodium_max",
    "glucose_min",
    "glucose_max",
    "urineoutput",
    "gcs_eyes",
    "gcs_verbal",
    "gcs_motor",
]
BASELINE_CLINICAL_COLUMNS = [
    "gender",
    "admission_age",
    "ethnicity",
    "admission_type",
    "admission_location",
    "marital_status",
    "gcs_unable",
    "sepsis3",
]
FEATURE_PRESET_CHOICES = [
    "all",
    "scores_only",
    "raw_signals_only",
    "clinical_core",
]
DEFAULT_SUMMARY_NUMERIC_COLUMNS = [
    "admission_age",
    "apsiii",
    "apsiii_prob",
    "heart_rate_max",
    "heart_rate_min",
    "mbp_min",
    "mbp_max",
    "temperature_max",
    "resp_rate_max",
    "creatinine_max",
    "bun_max",
    "sodium_min",
    "glucose_max",
    "urineoutput",
    "gcs_score",
]


@dataclass
class ExperimentResult:
    experiment_id: str
    method: str
    config: dict[str, object]
    fitted_model: object
    labels: np.ndarray
    metrics: dict[str, float]
    cluster_summary: pd.DataFrame
    assignments: pd.DataFrame
    profiles: pd.DataFrame


def log(message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara varios algoritmos de clustering sobre pacientes UCI con sepsis "
            "y genera salidas reutilizables para las siguientes capas del IDSS."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Ruta al CSV de entrada.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directorio donde se guardaran las salidas.",
    )
    parser.add_argument(
        "--target-col",
        default="hospital_expire_flag",
        help="Columna binaria opcional para resumir la mortalidad observada por cluster.",
    )
    parser.add_argument(
        "--categorical-cols",
        nargs="*",
        default=DEFAULT_CATEGORICAL_COLUMNS,
        help="Columnas categoricas que deben codificarse antes del clustering.",
    )
    parser.add_argument(
        "--id-cols",
        nargs="*",
        default=DEFAULT_ID_COLUMNS,
        help="Columnas identificadoras que se excluyen del clustering.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Algoritmos a comparar.",
    )
    parser.add_argument(
        "--feature-preset",
        default="all",
        choices=FEATURE_PRESET_CHOICES,
        help=(
            "Subconjunto de variables a usar. "
            "'scores_only' prioriza scores de gravedad; "
            "'raw_signals_only' prioriza senales fisiologicas y laboratorio; "
            "'clinical_core' mezcla variables clinicas clave y excluye leakage."
        ),
    )
    parser.add_argument("--min-k", type=int, default=3, help="Numero minimo de clusters.")
    parser.add_argument("--max-k", type=int, default=8, help="Numero maximo de clusters.")
    parser.add_argument(
        "--dbscan-eps",
        nargs="*",
        type=float,
        default=DEFAULT_DBSCAN_EPS,
        help="Valores de eps a probar para DBSCAN.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        nargs="*",
        type=int,
        default=DEFAULT_DBSCAN_MIN_SAMPLES,
        help="Valores de min_samples a probar para DBSCAN.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=15,
        help="Numero maximo de componentes para la reduccion dimensional.",
    )
    parser.add_argument(
        "--sample-size-metrics",
        type=int,
        default=5000,
        help="Numero maximo de observaciones usadas en las metricas internas.",
    )
    parser.add_argument(
        "--agglomerative-limit",
        type=int,
        default=12000,
        help="Maximo de filas para ejecutar clustering jerarquico aglomerativo.",
    )
    parser.add_argument("--profile-top-n", type=int, default=5)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Si se indica, limita el numero de filas usando una muestra aleatoria reproducible.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra el progreso detallado por terminal.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def sniff_csv_format(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()

    if header.count(";") > header.count(","):
        return ";", ","
    return ",", "."


def load_dataset(path: Path) -> pd.DataFrame:
    sep, decimal = sniff_csv_format(path)
    df = pd.read_csv(path, sep=sep, decimal=decimal)

    blank_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if blank_cols:
        df = df.rename(columns={blank_cols[0]: "Unnamed: 0"})

    return df


def maybe_sample_rows(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def apply_feature_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    if preset == "all":
        return df.drop(columns=[col for col in POST_OUTCOME_COLUMNS if col in df.columns], errors="ignore")

    keep_candidates: list[str] = []
    if preset == "scores_only":
        keep_candidates = BASELINE_CLINICAL_COLUMNS + SCORE_FAMILY_COLUMNS
    elif preset == "raw_signals_only":
        keep_candidates = BASELINE_CLINICAL_COLUMNS + ["apsiii", "gcs_score"] + RAW_SIGNAL_COLUMNS
    elif preset == "clinical_core":
        keep_candidates = (
            BASELINE_CLINICAL_COLUMNS
            + ["apsiii", "gcs_score"]
            + [
                "heart_rate_max",
                "heart_rate_min",
                "mbp_min",
                "mbp_max",
                "temperature_max",
                "resp_rate_max",
                "creatinine_max",
                "bun_max",
                "sodium_min",
                "glucose_max",
                "urineoutput",
            ]
        )

    keep_columns = [col for col in keep_candidates if col in df.columns]
    return df[keep_columns].copy()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def select_feature_columns(
    df: pd.DataFrame,
    target_col: str | None,
    categorical_cols: Iterable[str],
    id_cols: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    excluded = set(id_cols)
    if target_col and target_col in df.columns:
        excluded.add(target_col)

    candidate_cols = [col for col in df.columns if col not in excluded]
    valid_feature_cols: list[str] = []
    for col in candidate_cols:
        if df[col].nunique(dropna=False) <= 1:
            continue
        valid_feature_cols.append(col)

    categorical = [
        col for col in categorical_cols if col in valid_feature_cols and col in df.columns
    ]
    numeric = [col for col in valid_feature_cols if col not in categorical]
    return valid_feature_cols, numeric, categorical


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def reduce_dimensions(
    X: np.ndarray,
    max_components: int,
    random_state: int,
) -> tuple[np.ndarray, int, TruncatedSVD | None, StandardScaler]:
    if X.shape[1] <= 2:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        return scaled, X.shape[1], None, scaler

    n_components = min(max_components, X.shape[0] - 1, X.shape[1] - 1)
    if n_components < 2:
        n_components = min(2, X.shape[1])

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(X)
    scaler = StandardScaler()
    reduced = scaler.fit_transform(reduced)
    return reduced, n_components, svd, scaler


def sample_for_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= sample_size:
        return X, labels

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=sample_size, replace=False)
    return X[idx], labels[idx]


def filtered_cluster_labels(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = labels != -1
    if mask.sum() == 0:
        return np.empty((0, X.shape[1])), np.array([], dtype=int)
    return X[mask], labels[mask]


def count_clusters(labels: np.ndarray) -> int:
    return int(len(set(labels)) - (1 if -1 in labels else 0))


def evaluate_partition(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
) -> dict[str, float]:
    metrics = {
        "n_clusters": float(count_clusters(labels)),
        "noise_ratio": float(np.mean(labels == -1)),
        "coverage": float(np.mean(labels != -1)),
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
    }

    X_clustered, labels_clustered = filtered_cluster_labels(X, labels)
    if len(X_clustered) < 3 or len(np.unique(labels_clustered)) < 2:
        return metrics

    X_eval, labels_eval = sample_for_metrics(
        X_clustered,
        labels_clustered,
        sample_size=sample_size,
        random_state=random_state,
    )
    if len(np.unique(labels_eval)) < 2:
        return metrics

    try:
        metrics["silhouette"] = float(silhouette_score(X_eval, labels_eval))
    except Exception:
        pass

    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, labels_eval))
    except Exception:
        pass

    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_eval, labels_eval))
    except Exception:
        pass

    return metrics


def choose_severity_column(cluster_summary: pd.DataFrame) -> str | None:
    for candidate in ["mortality_rate", "mean_apsiii_prob", "mean_apsiii"]:
        if candidate in cluster_summary.columns:
            return candidate
    return None


def add_severity_rank(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    summary = cluster_summary.copy()
    severity_col = choose_severity_column(summary)
    if severity_col is None or summary.empty:
        summary["severity_rank"] = np.nan
        return summary

    real_clusters = summary[summary["cluster"] != -1].copy()
    if real_clusters.empty:
        summary["severity_rank"] = np.nan
        return summary

    real_clusters = real_clusters.sort_values(severity_col, ascending=False).reset_index(drop=True)
    real_clusters["severity_rank"] = np.arange(1, len(real_clusters) + 1)
    summary = summary.merge(
        real_clusters[["cluster", "severity_rank"]],
        on="cluster",
        how="left",
    )
    return summary


def safe_mode(series: pd.Series) -> object:
    mode = series.mode(dropna=True)
    if mode.empty:
        return np.nan
    return mode.iloc[0]


def cluster_mortality_dispersion(summary: pd.DataFrame) -> float:
    if "mortality_rate" not in summary.columns:
        return float("nan")
    real_clusters = summary.loc[summary["cluster"] != -1, "mortality_rate"]
    if real_clusters.empty:
        return float("nan")
    return float(real_clusters.std(ddof=0))


def build_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    target_col: str | None,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    summary_df = df.copy()
    summary_df["cluster"] = labels
    summary_df["cluster_name"] = np.where(summary_df["cluster"] == -1, "noise", "cluster")

    agg_kwargs: dict[str, tuple[str, str]] = {
        "n_patients": ("cluster", "size"),
    }
    if target_col and target_col in summary_df.columns:
        agg_kwargs["mortality_rate"] = (target_col, "mean")

    for col in DEFAULT_SUMMARY_NUMERIC_COLUMNS:
        if col in numeric_cols:
            agg_kwargs[f"mean_{col}"] = (col, "mean")

    for col in categorical_cols:
        agg_kwargs[f"mode_{col}"] = (col, safe_mode)

    cluster_summary = summary_df.groupby("cluster", as_index=False).agg(**agg_kwargs)
    cluster_summary["cluster_label"] = np.where(
        cluster_summary["cluster"] == -1,
        "noise",
        cluster_summary["cluster"].map(lambda value: f"cluster_{value}"),
    )
    cluster_summary["size_pct"] = cluster_summary["n_patients"] / len(df)
    cluster_summary = add_severity_rank(cluster_summary)
    sort_columns = ["severity_rank", "n_patients"]
    ascending = [True, False]
    if cluster_summary["severity_rank"].isna().all():
        sort_columns = ["n_patients"]
        ascending = [False]
    return cluster_summary.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def build_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_cols: list[str],
    top_n: int,
) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(columns=["cluster", "direction", "variable", "delta_vs_global"])

    numeric_df = df[numeric_cols].copy()
    global_means = numeric_df.mean(numeric_only=True)
    std = numeric_df.std(numeric_only=True).replace(0, np.nan)

    profile_rows: list[dict[str, object]] = []
    for cluster in sorted(pd.Series(labels).unique()):
        if cluster == -1:
            continue
        cluster_means = numeric_df[pd.Series(labels) == cluster].mean(numeric_only=True)
        standardized_delta = ((cluster_means - global_means) / std).dropna().sort_values()

        low_variables = standardized_delta.head(top_n)
        high_variables = standardized_delta.tail(top_n).sort_values(ascending=False)

        for variable, value in high_variables.items():
            profile_rows.append(
                {
                    "cluster": cluster,
                    "direction": "high",
                    "variable": variable,
                    "delta_vs_global": float(value),
                }
            )
        for variable, value in low_variables.items():
            profile_rows.append(
                {
                    "cluster": cluster,
                    "direction": "low",
                    "variable": variable,
                    "delta_vs_global": float(value),
                }
            )

    return pd.DataFrame(profile_rows)


def build_assignments(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_summary: pd.DataFrame,
    id_cols: list[str],
) -> pd.DataFrame:
    keep_cols = [col for col in id_cols if col in df.columns]
    assignments = df[keep_cols].copy() if keep_cols else pd.DataFrame(index=df.index)
    assignments["row_id"] = df.index
    assignments["cluster"] = labels
    assignments["cluster_label"] = np.where(assignments["cluster"] == -1, "noise", "cluster")
    assignments["cluster_label"] = np.where(
        assignments["cluster"] == -1,
        "noise",
        assignments["cluster"].map(lambda value: f"cluster_{value}"),
    )

    merge_cols = ["cluster"]
    if "severity_rank" in cluster_summary.columns:
        merge_cols.append("severity_rank")
    assignments = assignments.merge(
        cluster_summary[merge_cols],
        on="cluster",
        how="left",
    )
    return assignments


def density_config_id(config: dict[str, object]) -> str:
    parts = [f"{key}={value}" for key, value in sorted(config.items())]
    return ",".join(parts)


def metric_rank_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Convierte una metrica en una puntuacion percentil donde 1.0 siempre significa
    "mejor valor" y 0.0/"mas bajo" significa peor valor.
    """
    # Con pct=True, el rango mas alto recibe la puntuacion 1.0.
    # Por eso:
    # - si higher_is_better=True, queremos ordenar ascendentemente para que
    #   el valor mayor reciba el rango mas alto;
    # - si higher_is_better=False, queremos ordenar descendentemente para que
    #   el valor menor reciba el rango mas alto.
    return series.rank(ascending=higher_is_better, pct=True)


def estimate_total_experiments(
    methods: list[str],
    k_values: list[int],
    dbscan_eps_values: list[float],
    dbscan_min_samples_values: list[int],
    n_rows: int,
    agglomerative_limit: int,
) -> int:
    total = 0
    for method in methods:
        if method in {"kmeans", "gmm", "birch"}:
            total += len(k_values)
        elif method == "agglomerative":
            if n_rows <= agglomerative_limit:
                total += len(k_values)
        elif method == "dbscan":
            total += len(dbscan_eps_values) * len(dbscan_min_samples_values)
    return total


def rank_experiments(results: list[ExperimentResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        row: dict[str, object] = {
            "experiment_id": result.experiment_id,
            "method": result.method,
            "config": json.dumps(result.config, ensure_ascii=True, sort_keys=True),
        }
        row.update(result.metrics)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    comparison["rank_silhouette"] = metric_rank_score(
        comparison["silhouette"],
        higher_is_better=True,
    )
    comparison["rank_calinski"] = metric_rank_score(
        comparison["calinski_harabasz"],
        higher_is_better=True,
    )
    comparison["rank_davies"] = metric_rank_score(
        comparison["davies_bouldin"],
        higher_is_better=False,
    )
    comparison["rank_coverage"] = metric_rank_score(
        comparison["coverage"],
        higher_is_better=True,
    )
    comparison["rank_noise"] = metric_rank_score(
        comparison["noise_ratio"],
        higher_is_better=False,
    )

    rank_columns = [
        "rank_silhouette",
        "rank_calinski",
        "rank_davies",
        "rank_coverage",
        "rank_noise",
    ]
    if comparison["mortality_separation"].notna().any():
        comparison["rank_mortality_separation"] = metric_rank_score(
            comparison["mortality_separation"],
            higher_is_better=True,
        )
        rank_columns.append("rank_mortality_separation")

    comparison["selection_score"] = comparison[rank_columns].mean(axis=1, skipna=True)
    return comparison.sort_values(
        ["selection_score", "silhouette", "coverage"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def fit_models(
    X: np.ndarray,
    raw_df: pd.DataFrame,
    target_col: str | None,
    numeric_cols: list[str],
    categorical_cols: list[str],
    id_cols: list[str],
    methods: list[str],
    k_values: Iterable[int],
    dbscan_eps_values: list[float],
    dbscan_min_samples_values: list[int],
    sample_size_metrics: int,
    agglomerative_limit: int,
    profile_top_n: int,
    random_state: int,
    verbose: bool,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    completed = 0
    k_values_list = list(k_values)
    total_experiments = estimate_total_experiments(
        methods=methods,
        k_values=k_values_list,
        dbscan_eps_values=dbscan_eps_values,
        dbscan_min_samples_values=dbscan_min_samples_values,
        n_rows=len(raw_df),
        agglomerative_limit=agglomerative_limit,
    )

    log(f"[clustering] Experimentos previstos: {total_experiments}")

    def register_result(
        method: str,
        config: dict[str, object],
        fitted_model: object,
        labels: np.ndarray,
    ) -> None:
        if count_clusters(labels) < 2:
            if verbose:
                log(
                    f"[clustering] {method} {config} descartado: menos de 2 clusters utiles."
                )
            return

        cluster_summary = build_cluster_summary(
            df=raw_df,
            labels=labels,
            target_col=target_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        assignments = build_assignments(
            df=raw_df,
            labels=labels,
            cluster_summary=cluster_summary,
            id_cols=id_cols,
        )
        profiles = build_profiles(
            df=raw_df,
            labels=labels,
            numeric_cols=numeric_cols,
            top_n=profile_top_n,
        )
        metrics = evaluate_partition(
            X=X,
            labels=labels,
            sample_size=sample_size_metrics,
            random_state=random_state,
        )
        metrics["mortality_separation"] = cluster_mortality_dispersion(cluster_summary)
        experiment_id = f"{method}|{density_config_id(config)}"
        results.append(
            ExperimentResult(
                experiment_id=experiment_id,
                method=method,
                config=config,
                fitted_model=fitted_model,
                labels=labels,
                metrics=metrics,
                cluster_summary=cluster_summary,
                assignments=assignments,
                profiles=profiles,
            )
        )
        if verbose:
            silhouette = metrics["silhouette"]
            sil_text = "nan" if np.isnan(silhouette) else f"{silhouette:.4f}"
            log(
                f"[clustering] {method} {config} listo | "
                f"clusters={int(metrics['n_clusters'])} | "
                f"coverage={metrics['coverage']:.3f} | "
                f"silhouette={sil_text}"
            )

    for method in methods:
        if method in {"kmeans", "gmm", "birch", "agglomerative"}:
            for n_clusters in k_values_list:
                if method == "agglomerative" and len(X) > agglomerative_limit:
                    log(
                        f"[clustering] Agglomerative omitido para k={n_clusters} "
                        f"porque rows={len(X)} supera el limite {agglomerative_limit}."
                    )
                    continue

                config = {"n_clusters": n_clusters}
                completed += 1
                started_at = time.perf_counter()
                log(
                    f"[clustering] ({completed}/{total_experiments}) "
                    f"Ejecutando {method} {config}..."
                )
                if method == "gmm":
                    log(
                        "[clustering] GMM puede tardar bastante en bases grandes; "
                        "si solo quieres probar, usa --max-rows 15000."
                    )

                if method == "kmeans":
                    model = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        batch_size=1024,
                        n_init=20,
                        random_state=random_state,
                    )
                    labels = model.fit_predict(X)

                elif method == "gmm":
                    model = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type="full",
                        n_init=5,
                        random_state=random_state,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        labels = model.fit_predict(X)

                elif method == "birch":
                    model = Birch(n_clusters=n_clusters, threshold=0.5)
                    labels = model.fit_predict(X)

                else:
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
                    labels = model.fit_predict(X)

                register_result(
                    method=method,
                    config=config,
                    fitted_model=model,
                    labels=labels,
                )
                elapsed = time.perf_counter() - started_at
                log(f"[clustering] Tiempo {method} {config}: {elapsed:.1f}s")

        elif method == "dbscan":
            for eps in dbscan_eps_values:
                for min_samples in dbscan_min_samples_values:
                    config = {"eps": eps, "min_samples": min_samples}
                    completed += 1
                    started_at = time.perf_counter()
                    log(
                        f"[clustering] ({completed}/{total_experiments}) "
                        f"Ejecutando dbscan {config}..."
                    )
                    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                    labels = model.fit_predict(X)
                    register_result(
                        method=method,
                        config=config,
                        fitted_model=model,
                        labels=labels,
                    )
                    elapsed = time.perf_counter() - started_at
                    log(f"[clustering] Tiempo dbscan {config}: {elapsed:.1f}s")

    if not results:
        raise RuntimeError(
            "No se pudo generar ningun clustering valido. Revisa columnas, parametros o dependencias."
        )
    return results


def save_experiment_outputs(
    best_result: ExperimentResult,
    comparison: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, object],
    preprocessor: ColumnTransformer,
    svd: TruncatedSVD | None,
    scaler: StandardScaler,
    X_reduced: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    best_result.cluster_summary.to_csv(output_dir / "best_cluster_summary.csv", index=False)
    best_result.profiles.to_csv(output_dir / "best_cluster_profiles.csv", index=False)
    best_result.assignments.to_csv(output_dir / "best_patient_assignments.csv", index=False)

    embedding_columns = [f"dim_{idx + 1}" for idx in range(X_reduced.shape[1])]
    embeddings_df = pd.DataFrame(X_reduced, columns=embedding_columns)
    embeddings_df["row_id"] = np.arange(len(embeddings_df))
    embeddings_df = embeddings_df.merge(
        best_result.assignments[["row_id", "cluster", "cluster_label", "severity_rank"]],
        on="row_id",
        how="left",
    )
    embeddings_df.to_csv(output_dir / "reference_patient_embeddings.csv", index=False)

    joblib.dump(preprocessor, output_dir / "clustering_preprocessor.joblib")
    joblib.dump(svd, output_dir / "clustering_svd.joblib")
    joblib.dump(scaler, output_dir / "clustering_scaler.joblib")
    joblib.dump(best_result.fitted_model, output_dir / "clustering_model.joblib")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    log(f"[clustering] Cargando datos desde: {input_path}")
    df = load_dataset(input_path)
    log(f"[clustering] Dataset cargado: {len(df)} filas x {len(df.columns)} columnas")
    df = maybe_sample_rows(df=df, max_rows=args.max_rows, random_state=args.random_state)
    if args.max_rows is not None:
        log(f"[clustering] Filas usadas tras muestreo: {len(df)}")
    original_columns = list(df.columns)
    df = apply_feature_preset(df=df, preset=args.feature_preset)
    dropped_columns = [col for col in original_columns if col not in df.columns]
    log(
        f"[clustering] Preset de variables: {args.feature_preset} | "
        f"columnas usadas tras preset: {len(df.columns)}"
    )
    if dropped_columns:
        log(f"[clustering] Columnas excluidas por preset: {', '.join(dropped_columns)}")

    target_col = args.target_col if args.target_col in df.columns else None
    feature_cols, numeric_cols, categorical_cols = select_feature_columns(
        df=df,
        target_col=target_col,
        categorical_cols=args.categorical_cols,
        id_cols=args.id_cols,
    )
    if not numeric_cols and not categorical_cols:
        raise RuntimeError("No hay columnas validas para hacer clustering.")
    log(
        f"[clustering] Variables para clustering: "
        f"{len(feature_cols)} totales | {len(numeric_cols)} numericas | "
        f"{len(categorical_cols)} categoricas"
    )

    log("[clustering] Preparando matriz de entrada...")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_prepared = preprocessor.fit_transform(df[feature_cols])
    log(f"[clustering] Matriz preparada con forma: {X_prepared.shape}")

    log("[clustering] Aplicando reduccion dimensional...")
    X_reduced, n_components, svd, scaler = reduce_dimensions(
        X=X_prepared,
        max_components=args.svd_components,
        random_state=args.random_state,
    )
    log(f"[clustering] Matriz reducida a {n_components} componentes")

    k_values = list(range(args.min_k, args.max_k + 1))
    log(
        f"[clustering] Metodos solicitados: {', '.join(args.methods)} | "
        f"k={k_values}"
    )
    results = fit_models(
        X=X_reduced,
        raw_df=df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        id_cols=args.id_cols,
        methods=args.methods,
        k_values=k_values,
        dbscan_eps_values=args.dbscan_eps,
        dbscan_min_samples_values=args.dbscan_min_samples,
        sample_size_metrics=args.sample_size_metrics,
        agglomerative_limit=args.agglomerative_limit,
        profile_top_n=args.profile_top_n,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    log("[clustering] Rankeando experimentos...")
    comparison = rank_experiments(results)
    best_result = next(
        result for result in results if result.experiment_id == comparison.iloc[0]["experiment_id"]
    )

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir.resolve()),
        "rows": int(len(df)),
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "methods": args.methods,
        "k_values": list(k_values),
        "dbscan_eps": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
        "selected_model": {
            "method": best_result.method,
            "config": best_result.config,
        },
        "reduced_components": n_components,
        "artifacts": {
            "preprocessor": "clustering_preprocessor.joblib",
            "svd": "clustering_svd.joblib",
            "scaler": "clustering_scaler.joblib",
            "model": "clustering_model.joblib",
            "reference_embeddings": "reference_patient_embeddings.csv",
        },
    }
    log(f"[clustering] Guardando resultados en: {output_dir}")
    save_experiment_outputs(
        best_result,
        comparison,
        output_dir,
        metadata,
        preprocessor=preprocessor,
        svd=svd,
        scaler=scaler,
        X_reduced=X_reduced,
    )

    print("Mejor experimento de clustering:", flush=True)
    print(f"- metodo: {best_result.method}", flush=True)
    print(f"- configuracion: {best_result.config}", flush=True)
    print("- metricas:", flush=True)
    for metric_name, metric_value in best_result.metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}", flush=True)
    print(f"Salidas guardadas en: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
