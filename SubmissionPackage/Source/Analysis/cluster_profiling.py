from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal


DEFAULT_CLUSTERING_DIR = Path("outputs") / "clustering_clinical"
DEFAULT_XGB_DIR = Path("outputs") / "xgboost_explainability"
DEFAULT_DATA_PATH = Path("bbdd") / "BaseDatos_imputada_knn.csv"
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
DEFAULT_INVERT_VARS = {
    "gcs_motor",
    "gcs_verbal",
    "gcs_eyes",
    "urineoutput",
    "mbp_min",
}
CV_LOW_UNCERTAINTY = 0.15
CV_HIGH_UNCERTAINTY = 0.50
LOW_THRESH = 33
HIGH_THRESH = 66
COLOR_GREEN = np.array([0.18, 0.70, 0.30])
COLOR_YELLOW = np.array([0.95, 0.78, 0.10])
COLOR_RED = np.array([0.85, 0.20, 0.15])
COLOR_GRAY = np.array([0.78, 0.78, 0.78])

RENAL_FEATURES = {
    "creatinine_max",
    "creatinine_min",
    "creatinine_score",
    "bun_max",
    "bun_min",
    "bun_score",
    "urineoutput",
    "uo_score",
    "sodium_min",
    "sodium_max",
    "glucose_max",
    "glucose_min",
}
NEURO_FEATURES = {"gcs_score", "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable"}
INFLAMMATORY_FEATURES = {
    "temp_score",
    "temperature_max",
    "temperature_min",
    "wbc_score",
    "wbc_max",
    "wbc_min",
    "resp_rate_score",
    "resp_rate_max",
    "resp_rate_min",
    "heart_rate_max",
    "heart_rate_min",
    "hr_score",
}
HEMODYNAMIC_FEATURES = {"mbp_score", "mbp_min", "mbp_max"}
SEVERITY_FEATURES = {"apsiii", "apsiii_prob", "sepsis3", "admission_age"}


@dataclass
class VariableGroups:
    numeric: list[str]
    categorical: list[str]
    binary: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Amplia el profiling del clustering con una capa estilo aTLP, "
            "validacion estadistica y salidas listas para documentacion."
        )
    )
    parser.add_argument("--clustering-dir", default=str(DEFAULT_CLUSTERING_DIR))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--xgb-dir", default=str(DEFAULT_XGB_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--categorical-cols",
        nargs="*",
        default=DEFAULT_CATEGORICAL_COLUMNS,
    )
    parser.add_argument(
        "--id-cols",
        nargs="*",
        default=DEFAULT_ID_COLUMNS,
    )
    parser.add_argument(
        "--invert-vars",
        nargs="*",
        default=sorted(DEFAULT_INVERT_VARS),
    )
    parser.add_argument("--top-n-features", type=int, default=5)
    parser.add_argument("--max-atlp-vars", type=int, default=18)
    parser.add_argument("--low-thresh", type=float, default=LOW_THRESH)
    parser.add_argument("--high-thresh", type=float, default=HIGH_THRESH)
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def log(message: str) -> None:
    print(f"[profiling] {message}", flush=True)


def sniff_csv_format(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()
    if header.count(";") > header.count(","):
        return ";", ","
    return ",", "."


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero requerido: {path}")
    return pd.read_csv(path)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset: {path}")
    sep, decimal = sniff_csv_format(path)
    df = pd.read_csv(path, sep=sep, decimal=decimal)
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def sanitize_cluster_label(label: object, cluster: object) -> str:
    if pd.notna(label):
        return str(label)
    return f"cluster_{cluster}"


def collect_feature_list(
    profiles_df: pd.DataFrame,
    cluster: int,
    direction: str,
    top_n: int,
) -> list[str]:
    filtered = profiles_df[
        (profiles_df["cluster"] == cluster) & (profiles_df["direction"] == direction)
    ].copy()
    if filtered.empty:
        return []

    sort_ascending = direction == "low"
    filtered = filtered.sort_values("delta_vs_global", ascending=sort_ascending)
    return filtered["variable"].head(top_n).tolist()


def format_feature_values(
    profiles_df: pd.DataFrame,
    cluster: int,
    direction: str,
    top_n: int,
) -> str:
    filtered = profiles_df[
        (profiles_df["cluster"] == cluster) & (profiles_df["direction"] == direction)
    ].copy()
    if filtered.empty:
        return "none"

    sort_ascending = direction == "low"
    filtered = filtered.sort_values("delta_vs_global", ascending=sort_ascending).head(top_n)
    return ", ".join(
        f"{row.variable} ({row.delta_vs_global:+.3f})" for row in filtered.itertuples()
    )


def feature_family_votes(features: list[str]) -> dict[str, int]:
    votes = {
        "renal_metabolic": 0,
        "neurologic": 0,
        "inflammatory": 0,
        "hemodynamic": 0,
        "severity": 0,
        "frailty": 0,
    }
    for feature in features:
        if feature in RENAL_FEATURES:
            votes["renal_metabolic"] += 1
        if feature in NEURO_FEATURES:
            votes["neurologic"] += 1
        if feature in INFLAMMATORY_FEATURES:
            votes["inflammatory"] += 1
        if feature in HEMODYNAMIC_FEATURES:
            votes["hemodynamic"] += 1
        if feature in SEVERITY_FEATURES:
            votes["severity"] += 1
        if feature == "admission_age":
            votes["frailty"] += 2
    return votes


def suggest_phenotype_name(
    high_features: list[str],
    low_features: list[str],
    severity_rank: float | int | None,
    summary_row: pd.Series | None = None,
) -> str:
    votes = feature_family_votes(high_features)
    low_feature_set = set(low_features)
    summary_row = summary_row if summary_row is not None else pd.Series(dtype=object)

    mean_age = extract_mean_value(summary_row, "admission_age")
    mean_apsiii = extract_mean_value(summary_row, "apsiii")
    mean_creatinine = extract_mean_value(summary_row, "creatinine_max")
    mean_bun = extract_mean_value(summary_row, "bun_max")
    mean_temp = extract_mean_value(summary_row, "temperature_max")
    mean_gcs_score = extract_mean_value(summary_row, "gcs_score")

    if severity_rank is not None and pd.notna(severity_rank):
        severity_rank = int(severity_rank)
    else:
        severity_rank = None

    if (
        severity_rank is not None
        and severity_rank >= 4
        and (mean_apsiii is not None and mean_apsiii < 35)
    ):
        if {"apsiii", "creatinine_max", "bun_max", "sepsis3", "gcs_score"}.intersection(low_feature_set):
            return "stable_low_severity"

    if votes["renal_metabolic"] >= 2 and votes["severity"] >= 1:
        return "renal_metabolic_high_severity"
    if (
        votes["neurologic"] >= 1
        and votes["inflammatory"] >= 1
        and votes["severity"] >= 1
        and (
            (mean_gcs_score is not None and mean_gcs_score >= 10)
            or severity_rank in {1, 2}
            or (mean_temp is not None and mean_temp >= 37.4)
        )
    ):
        return "neurologic_inflammatory_high_severity"
    if (
        mean_age is not None
        and mean_age >= 70
        and (mean_apsiii is not None and mean_apsiii < 45)
    ):
        return "older_frail_lower_severity"
    if (
        mean_age is not None
        and mean_age >= 67
        and (
            (mean_apsiii is not None and 45 <= mean_apsiii < 55)
            or (mean_bun is not None and mean_bun >= 25)
            or (mean_creatinine is not None and mean_creatinine >= 1.2)
        )
    ):
        return "older_intermediate_moderate_severity"
    if votes["hemodynamic"] >= 2 and votes["severity"] >= 1:
        return "hemodynamic_instability"
    if severity_rank is not None and severity_rank >= 5:
        return "stable_low_severity"
    if votes["severity"] >= 2:
        return "moderate_to_high_severity"
    if severity_rank is not None and severity_rank >= 4:
        return "mixed_lower_severity"
    return "mixed_intermediate"


def extract_mean_value(row: pd.Series, feature: str) -> float | None:
    column_name = f"mean_{feature}"
    if column_name not in row.index or pd.isna(row[column_name]):
        return None
    return float(row[column_name])


def build_clinical_snapshot(row: pd.Series) -> str:
    pieces: list[str] = []
    for feature in [
        "apsiii",
        "admission_age",
        "creatinine_max",
        "bun_max",
        "mbp_min",
        "temperature_max",
        "urineoutput",
        "gcs_score",
    ]:
        value = extract_mean_value(row, feature)
        if value is None:
            continue
        pieces.append(f"{feature}={value:.2f}")
    return ", ".join(pieces)


def load_xgb_cluster_summary(xgb_dir: Path) -> pd.DataFrame | None:
    path = xgb_dir / "cluster_explainability_summary.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    rename_map = {
        "n_patients_test": "xgb_n_patients_test",
        "observed_mortality_test": "xgb_observed_mortality_test",
        "mean_predicted_probability": "xgb_mean_predicted_probability",
        "top_explanatory_features": "xgb_top_explanatory_features",
    }
    return df.rename(columns=rename_map)


def build_profile_table(
    cluster_summary: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    xgb_cluster_summary: pd.DataFrame | None,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    xgb_lookup = None
    if xgb_cluster_summary is not None:
        xgb_lookup = xgb_cluster_summary.set_index("cluster_label")

    for row in cluster_summary.itertuples(index=False):
        cluster_label = sanitize_cluster_label(getattr(row, "cluster_label", np.nan), row.cluster)
        high_features = collect_feature_list(cluster_profiles, row.cluster, "high", top_n)
        low_features = collect_feature_list(cluster_profiles, row.cluster, "low", top_n)
        phenotype_name = suggest_phenotype_name(
            high_features=high_features,
            low_features=low_features,
            severity_rank=getattr(row, "severity_rank", np.nan),
            summary_row=pd.Series(row._asdict()),
        )

        profile_row: dict[str, object] = {
            "cluster": row.cluster,
            "cluster_label": cluster_label,
            "phenotype_name": phenotype_name,
            "severity_rank": getattr(row, "severity_rank", np.nan),
            "n_patients": getattr(row, "n_patients", np.nan),
            "size_pct": getattr(row, "size_pct", np.nan),
            "top_high_features": format_feature_values(cluster_profiles, row.cluster, "high", top_n),
            "top_low_features": format_feature_values(cluster_profiles, row.cluster, "low", top_n),
            "clinical_snapshot": build_clinical_snapshot(pd.Series(row._asdict())),
        }

        if "mortality_rate" in cluster_summary.columns:
            profile_row["clustering_mortality_rate"] = getattr(row, "mortality_rate", np.nan)

        if xgb_lookup is not None and cluster_label in xgb_lookup.index:
            xgb_row = xgb_lookup.loc[cluster_label]
            if isinstance(xgb_row, pd.DataFrame):
                xgb_row = xgb_row.iloc[0]
            profile_row["xgb_n_patients_test"] = xgb_row.get("xgb_n_patients_test", np.nan)
            profile_row["xgb_observed_mortality_test"] = xgb_row.get("xgb_observed_mortality_test", np.nan)
            profile_row["xgb_mean_predicted_probability"] = xgb_row.get("xgb_mean_predicted_probability", np.nan)
            profile_row["xgb_top_explanatory_features"] = xgb_row.get("xgb_top_explanatory_features", "")

        rows.append(profile_row)

    profiling_df = pd.DataFrame(rows)
    return profiling_df.sort_values("severity_rank", na_position="last").reset_index(drop=True)


def build_report(profiling_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("Cluster Profiling Report")
    lines.append("=======================")
    lines.append("")
    lines.append("Este informe resume cada cluster como fenotipo clinico reutilizable para el IDSS.")
    lines.append("")

    for row in profiling_df.itertuples(index=False):
        title = f"{row.cluster_label} | {row.phenotype_name}"
        lines.append(title)
        lines.append("-" * max(12, len(title)))
        lines.append(f"Severity rank: {row.severity_rank}")
        lines.append(f"Patients: {row.n_patients} ({float(row.size_pct) * 100:.2f}%)")
        if "clustering_mortality_rate" in profiling_df.columns and pd.notna(getattr(row, "clustering_mortality_rate", np.nan)):
            lines.append(f"Observed mortality in clustering cohort: {float(row.clustering_mortality_rate):.4f}")
        if pd.notna(getattr(row, "xgb_mean_predicted_probability", np.nan)):
            lines.append(
                "XGBoost context: "
                f"mean predicted probability={float(row.xgb_mean_predicted_probability):.4f}, "
                f"observed mortality test={float(row.xgb_observed_mortality_test):.4f}"
            )
        lines.append(f"Clinical snapshot: {row.clinical_snapshot}")
        lines.append(f"Top high features: {row.top_high_features}")
        lines.append(f"Top low features: {row.top_low_features}")
        if isinstance(getattr(row, "xgb_top_explanatory_features", ""), str) and row.xgb_top_explanatory_features:
            lines.append(f"Main XGBoost explanatory features: {row.xgb_top_explanatory_features}")
        lines.append("")

    return "\n".join(lines)


def merge_cluster_assignments(raw_df: pd.DataFrame, assignments_df: pd.DataFrame) -> pd.DataFrame:
    merged = raw_df.reset_index().rename(columns={"index": "row_id"}).merge(
        assignments_df,
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("La union entre dataset y asignaciones ha quedado vacia.")
    return merged


def candidate_atlp_variables(
    raw_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    categorical_cols: list[str],
    id_cols: list[str],
    max_atlp_vars: int,
) -> list[str]:
    summary_vars = []
    for column in cluster_summary.columns:
        if column.startswith("mean_"):
            source = column.replace("mean_", "", 1)
            if source in raw_df.columns:
                summary_vars.append(source)

    profile_rank = (
        cluster_profiles["variable"].value_counts().sort_values(ascending=False).index.tolist()
        if not cluster_profiles.empty
        else []
    )

    ordered: list[str] = []
    seen: set[str] = set()
    for variable in summary_vars + profile_rank:
        if variable in seen or variable not in raw_df.columns:
            continue
        if variable in categorical_cols or variable in id_cols:
            continue
        ordered.append(variable)
        seen.add(variable)
        if len(ordered) >= max_atlp_vars:
            break
    return ordered


def infer_variable_groups(
    df: pd.DataFrame,
    candidate_vars: list[str],
    categorical_cols: list[str],
) -> VariableGroups:
    numeric: list[str] = []
    categorical: list[str] = []
    binary: list[str] = []

    categorical_set = set(categorical_cols)
    for variable in candidate_vars:
        if variable not in df.columns:
            continue

        series = df[variable].dropna()
        if variable in categorical_set:
            categorical.append(variable)
            continue

        if pd.api.types.is_bool_dtype(df[variable]):
            binary.append(variable)
            continue

        if pd.api.types.is_numeric_dtype(df[variable]):
            unique_values = set(series.unique().tolist())
            if unique_values.issubset({0, 1}) and unique_values:
                binary.append(variable)
            else:
                numeric.append(variable)
            continue

        categorical.append(variable)

    for variable in categorical_cols:
        if variable in df.columns and variable not in categorical:
            categorical.append(variable)

    return VariableGroups(numeric=numeric, categorical=categorical, binary=binary)


def get_tlp_color_label(value: float, p_low: float, p_high: float, invert: bool) -> str:
    if pd.isna(value) or pd.isna(p_low) or pd.isna(p_high):
        return "yellow"
    if invert:
        if value >= p_high:
            return "green"
        if value <= p_low:
            return "red"
        return "yellow"
    if value <= p_low:
        return "green"
    if value >= p_high:
        return "red"
    return "yellow"


def compute_numeric_uncertainty(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return 1.0
    mean = float(series.mean())
    std = float(series.std())
    median = float(series.median())
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    unique_count = int(series.nunique())

    cv = abs(std / mean) if abs(mean) >= 1e-9 else 1.0
    robust_dispersion = iqr / (abs(median) + 1e-9) if abs(median) >= 1e-9 else cv

    if unique_count <= 20:
        span = float(series.max() - series.min())
        bounded_dispersion = iqr / span if span > 1e-9 else 0.0
        uncertainty = 0.4 * np.clip(cv, 0.0, 1.0) + 0.35 * np.clip(robust_dispersion, 0.0, 1.0) + 0.25 * np.clip(bounded_dispersion, 0.0, 1.0)
    else:
        uncertainty = min(np.clip(cv, 0.0, 1.0), np.clip(robust_dispersion, 0.0, 1.0))

    return float(np.clip(uncertainty, 0.0, 1.0))


def compute_binary_uncertainty(series: pd.Series) -> float:
    probability = float(series.mean())
    return float(1.0 - abs(2 * probability - 1))


def compute_normalized_entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    entropy = -np.sum(counts * np.log2(counts + 1e-12))
    max_entropy = np.log2(len(counts))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def uncertainty_to_confidence(uncertainty: float) -> str:
    if uncertainty < CV_LOW_UNCERTAINTY:
        return "high"
    if uncertainty > CV_HIGH_UNCERTAINTY:
        return "low"
    return "medium"


def color_label_to_rgb(label: str) -> np.ndarray:
    return {
        "green": COLOR_GREEN,
        "yellow": COLOR_YELLOW,
        "red": COLOR_RED,
    }.get(label, COLOR_YELLOW)


def blend_with_gray(base_color: np.ndarray, uncertainty: float) -> np.ndarray:
    saturation = 1.0 - np.clip(float(uncertainty), 0.0, 1.0) * 0.85
    return base_color * saturation + COLOR_GRAY * (1.0 - saturation)


def build_atlp_summary(
    clustered_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    variable_groups: VariableGroups,
    invert_vars: set[str],
    low_thresh: float,
    high_thresh: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = cluster_summary[["cluster", "cluster_label", "severity_rank"]].copy()
    labels["cluster_label"] = labels.apply(
        lambda row: sanitize_cluster_label(row["cluster_label"], row["cluster"]),
        axis=1,
    )

    numeric_like_vars = variable_groups.numeric + variable_groups.binary
    global_stats: dict[str, dict[str, float]] = {}
    for variable in numeric_like_vars:
        series = pd.to_numeric(clustered_df[variable], errors="coerce").dropna()
        global_stats[variable] = {
            "median": float(series.median()) if not series.empty else np.nan,
            "mean": float(series.mean()) if not series.empty else np.nan,
            "p_low": float(np.nanpercentile(series, low_thresh)) if not series.empty else np.nan,
            "p_high": float(np.nanpercentile(series, high_thresh)) if not series.empty else np.nan,
            "std": float(series.std()) if not series.empty else np.nan,
        }

    for variable in variable_groups.categorical:
        if variable not in clustered_df.columns:
            continue
        series = clustered_df[variable].astype(str)
        global_mode = series.mode(dropna=True)
        global_stats[variable] = {
            "global_mode": str(global_mode.iloc[0]) if not global_mode.empty else "",
            "global_mode_share": float((series == global_mode.iloc[0]).mean()) if not global_mode.empty else np.nan,
        }

    for cluster_row in labels.sort_values("severity_rank", na_position="last").itertuples(index=False):
        subset = clustered_df[clustered_df["cluster"] == cluster_row.cluster].copy()
        if subset.empty:
            continue

        for variable in variable_groups.numeric:
            series = pd.to_numeric(subset[variable], errors="coerce").dropna()
            prototype = float(series.median()) if not series.empty else np.nan
            stats = global_stats[variable]
            uncertainty = compute_numeric_uncertainty(series) if not series.empty else 1.0
            color = get_tlp_color_label(
                value=prototype,
                p_low=stats["p_low"],
                p_high=stats["p_high"],
                invert=variable in invert_vars,
            )
            std = stats["std"] if not pd.isna(stats["std"]) else np.nan
            z_score = (prototype - stats["mean"]) / std if std and not pd.isna(std) else np.nan
            rows.append(
                {
                    "cluster": cluster_row.cluster,
                    "cluster_label": cluster_row.cluster_label,
                    "severity_rank": cluster_row.severity_rank,
                    "variable": variable,
                    "variable_type": "numeric",
                    "prototype": prototype,
                    "global_reference": stats["median"],
                    "delta_vs_global": prototype - stats["median"] if not pd.isna(prototype) else np.nan,
                    "z_score_vs_global": z_score,
                    "tlp_color": color,
                    "uncertainty": uncertainty,
                    "confidence": uncertainty_to_confidence(uncertainty),
                }
            )

        for variable in variable_groups.binary:
            series = pd.to_numeric(subset[variable], errors="coerce").dropna()
            prototype = float(series.mean()) if not series.empty else np.nan
            stats = global_stats[variable]
            uncertainty = compute_binary_uncertainty(series) if not series.empty else 1.0
            color = get_tlp_color_label(
                value=prototype,
                p_low=stats["p_low"],
                p_high=stats["p_high"],
                invert=variable in invert_vars,
            )
            rows.append(
                {
                    "cluster": cluster_row.cluster,
                    "cluster_label": cluster_row.cluster_label,
                    "severity_rank": cluster_row.severity_rank,
                    "variable": variable,
                    "variable_type": "binary",
                    "prototype": prototype,
                    "global_reference": stats["mean"],
                    "delta_vs_global": prototype - stats["mean"] if not pd.isna(prototype) else np.nan,
                    "z_score_vs_global": np.nan,
                    "tlp_color": color,
                    "uncertainty": uncertainty,
                    "confidence": uncertainty_to_confidence(uncertainty),
                }
            )

        for variable in variable_groups.categorical:
            series = subset[variable].astype(str)
            mode = series.mode(dropna=True)
            dominant = str(mode.iloc[0]) if not mode.empty else ""
            dominant_share = float((series == dominant).mean()) if dominant else np.nan
            uncertainty = compute_normalized_entropy(series)
            global_mode = global_stats[variable]["global_mode"]
            global_share = global_stats[variable]["global_mode_share"]
            rows.append(
                {
                    "cluster": cluster_row.cluster,
                    "cluster_label": cluster_row.cluster_label,
                    "severity_rank": cluster_row.severity_rank,
                    "variable": variable,
                    "variable_type": "categorical",
                    "prototype": dominant,
                    "global_reference": global_mode,
                    "delta_vs_global": dominant_share - global_share if pd.notna(dominant_share) and pd.notna(global_share) else np.nan,
                    "z_score_vs_global": np.nan,
                    "tlp_color": "yellow",
                    "uncertainty": uncertainty,
                    "confidence": uncertainty_to_confidence(uncertainty),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["severity_rank", "cluster", "variable_type", "variable"],
        na_position="last",
    ).reset_index(drop=True)


def build_categorical_context_summary(atlp_summary: pd.DataFrame) -> pd.DataFrame:
    categorical = atlp_summary[atlp_summary["variable_type"] == "categorical"].copy()
    if categorical.empty:
        return categorical
    return categorical[
        [
            "cluster",
            "cluster_label",
            "severity_rank",
            "variable",
            "prototype",
            "global_reference",
            "delta_vs_global",
            "uncertainty",
            "confidence",
        ]
    ].rename(
        columns={
            "prototype": "dominant_category",
            "global_reference": "global_mode",
            "delta_vs_global": "dominant_share_delta",
        }
    )


def build_numeric_association_table(clustered_df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    k = clustered_df["cluster"].nunique()
    n = len(clustered_df)
    for variable in variables:
        groups = []
        for _, subset in clustered_df.groupby("cluster"):
            series = pd.to_numeric(subset[variable], errors="coerce").dropna()
            if not series.empty:
                groups.append(series.values)
        if len(groups) < 2:
            continue
        stat, p_value = kruskal(*groups)
        eta_sq = max((stat - k + 1) / (n - k), 0.0) if n > k else np.nan
        rows.append(
            {
                "variable": variable,
                "kruskal_h": float(stat),
                "p_value": float(p_value),
                "eta_squared": float(eta_sq),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["eta_squared", "kruskal_h"], ascending=[False, False]
    ).reset_index(drop=True)


def cramers_v(confusion: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.values.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def build_categorical_association_table(clustered_df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in variables:
        table = pd.crosstab(clustered_df["cluster"], clustered_df[variable].astype(str))
        if table.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = chi2_contingency(table)
        rows.append(
            {
                "variable": variable,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "cramers_v": cramers_v(table),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cramers_v", "chi2"], ascending=[False, False]
    ).reset_index(drop=True)


def pivot_atlp_numeric(atlp_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_df = atlp_summary[atlp_summary["variable_type"].isin(["numeric", "binary"])].copy()
    order = (
        numeric_df[["cluster_label", "severity_rank"]]
        .drop_duplicates()
        .sort_values(["severity_rank", "cluster_label"], na_position="last")["cluster_label"]
        .tolist()
    )
    color_pivot = numeric_df.pivot(index="cluster_label", columns="variable", values="tlp_color")
    prototype_pivot = numeric_df.pivot(index="cluster_label", columns="variable", values="prototype")
    uncertainty_pivot = numeric_df.pivot(index="cluster_label", columns="variable", values="uncertainty")
    return color_pivot.reindex(order), prototype_pivot.reindex(order), uncertainty_pivot.reindex(order)


def plot_atlp_panel(atlp_summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    from matplotlib import pyplot as plt

    matplotlib.use("Agg")
    color_pivot, prototype_pivot, uncertainty_pivot = pivot_atlp_numeric(atlp_summary)
    if color_pivot.empty:
        return

    rows = len(color_pivot.index)
    cols = len(color_pivot.columns)
    rgb = np.zeros((rows, cols, 3))
    for i, _ in enumerate(color_pivot.index):
        for j, _ in enumerate(color_pivot.columns):
            color = color_pivot.iloc[i, j]
            uncertainty = uncertainty_pivot.iloc[i, j]
            rgb[i, j, :] = blend_with_gray(color_label_to_rgb(str(color)), float(uncertainty))

    fig_width = max(12, cols * 0.8)
    fig_height = max(4, rows * 0.9 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(rgb, aspect="auto")

    for i, _ in enumerate(color_pivot.index):
        for j, _ in enumerate(color_pivot.columns):
            value = prototype_pivot.iloc[i, j]
            if pd.isna(value):
                label = ""
            elif float(value).is_integer():
                label = str(int(value))
            elif abs(float(value)) >= 100:
                label = f"{float(value):.0f}"
            else:
                label = f"{float(value):.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")

    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(color_pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(color_pivot.index, fontsize=10)
    ax.set_title("aTLP Clinical Profiling Panel", fontweight="bold")
    ax.set_xlabel("Clinical variables")
    ax.set_ylabel("Clusters")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_GREEN, label="Green: low or favorable"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_YELLOW, label="Yellow: intermediate"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RED, label="Red: high or unfavorable"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_GRAY, label="Gray tint: high uncertainty"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_heatmap(atlp_summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    import seaborn as sns
    from matplotlib import pyplot as plt

    matplotlib.use("Agg")
    _, _, uncertainty_pivot = pivot_atlp_numeric(atlp_summary)
    if uncertainty_pivot.empty:
        return

    fig_width = max(12, len(uncertainty_pivot.columns) * 0.7)
    fig_height = max(4, len(uncertainty_pivot.index) * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        uncertainty_pivot.astype(float),
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Uncertainty"},
        ax=ax,
    )
    ax.set_title("aTLP Uncertainty Heatmap", fontweight="bold")
    ax.set_xlabel("Clinical variables")
    ax.set_ylabel("Clusters")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_validation_report(
    profile_table: pd.DataFrame,
    atlp_summary: pd.DataFrame,
    numeric_tests: pd.DataFrame,
    categorical_tests: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Cluster Profiling Validation")
    lines.append("")
    lines.append("This report extends the base cluster profiling with an aTLP-style summary.")
    lines.append("")

    if not numeric_tests.empty:
        lines.append("## Most discriminative numeric variables")
        lines.append("")
        lines.append("| Variable | Kruskal H | p-value | eta squared |")
        lines.append("|---|---:|---:|---:|")
        for row in numeric_tests.head(12).itertuples(index=False):
            lines.append(
                f"| {row.variable} | {row.kruskal_h:.2f} | {row.p_value:.3e} | {row.eta_squared:.3f} |"
            )
        lines.append("")

    if not categorical_tests.empty:
        lines.append("## Most discriminative categorical variables")
        lines.append("")
        lines.append("| Variable | Chi2 | p-value | Cramers V |")
        lines.append("|---|---:|---:|---:|")
        for row in categorical_tests.head(12).itertuples(index=False):
            lines.append(
                f"| {row.variable} | {row.chi2:.2f} | {row.p_value:.3e} | {row.cramers_v:.3f} |"
            )
        lines.append("")

    lines.append("## Cluster-level interpretation")
    lines.append("")
    for cluster_row in profile_table.itertuples(index=False):
        lines.append(f"### {cluster_row.cluster_label} | {cluster_row.phenotype_name}")
        lines.append("")
        lines.append(f"- Severity rank: {cluster_row.severity_rank}")
        lines.append(f"- Size: {cluster_row.n_patients} patients ({float(cluster_row.size_pct) * 100:.2f}%)")
        lines.append(f"- Snapshot: {cluster_row.clinical_snapshot}")
        lines.append(f"- Top high features: {cluster_row.top_high_features}")
        lines.append(f"- Top low features: {cluster_row.top_low_features}")

        subset = atlp_summary[
            (atlp_summary["cluster"] == cluster_row.cluster)
            & (atlp_summary["variable_type"].isin(["numeric", "binary"]))
        ].copy()
        salient = subset.sort_values("z_score_vs_global", ascending=False).head(3)
        suppressed = subset.sort_values("z_score_vs_global", ascending=True).head(3)
        uncertain = subset.sort_values("uncertainty", ascending=False).head(3)

        if not salient.empty:
            lines.append(
                "- Salient high profile: "
                + ", ".join(
                    f"{row.variable} ({row.prototype:.2f}, {row.tlp_color}, {row.confidence})"
                    for row in salient.itertuples(index=False)
                    if pd.notna(row.prototype)
                )
            )
        if not suppressed.empty:
            lines.append(
                "- Salient low profile: "
                + ", ".join(
                    f"{row.variable} ({row.prototype:.2f}, {row.tlp_color}, {row.confidence})"
                    for row in suppressed.itertuples(index=False)
                    if pd.notna(row.prototype)
                )
            )
        if not uncertain.empty:
            lines.append(
                "- Most uncertain cells: "
                + ", ".join(
                    f"{row.variable} (U={row.uncertainty:.2f})"
                    for row in uncertain.itertuples(index=False)
                )
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()

    clustering_dir = Path(args.clustering_dir)
    xgb_dir = Path(args.xgb_dir)
    output_dir = Path(args.output_dir) if args.output_dir else clustering_dir / "profiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Cargando clustering desde: {clustering_dir}")
    cluster_summary = load_csv(clustering_dir / "best_cluster_summary.csv")
    cluster_profiles = load_csv(clustering_dir / "best_cluster_profiles.csv")
    assignments = load_csv(clustering_dir / "best_patient_assignments.csv")
    raw_df = load_dataset(Path(args.data_path))
    xgb_cluster_summary = load_xgb_cluster_summary(xgb_dir)

    log("Preparando tabla base de profiling...")
    profile_table = build_profile_table(
        cluster_summary=cluster_summary,
        cluster_profiles=cluster_profiles,
        xgb_cluster_summary=xgb_cluster_summary,
        top_n=args.top_n_features,
    )
    profile_table.to_csv(output_dir / "cluster_profile_table.csv", index=False)
    (output_dir / "cluster_profile_report.txt").write_text(
        build_report(profile_table),
        encoding="utf-8",
    )

    log("Reconstruyendo cohorte etiquetada por cluster...")
    clustered_df = merge_cluster_assignments(raw_df=raw_df, assignments_df=assignments)
    candidate_vars = candidate_atlp_variables(
        raw_df=clustered_df,
        cluster_summary=cluster_summary,
        cluster_profiles=cluster_profiles,
        categorical_cols=args.categorical_cols,
        id_cols=args.id_cols,
        max_atlp_vars=args.max_atlp_vars,
    )
    variable_groups = infer_variable_groups(
        df=clustered_df,
        candidate_vars=candidate_vars,
        categorical_cols=args.categorical_cols,
    )

    log(
        "Variables para aTLP: "
        f"{len(variable_groups.numeric)} numericas, "
        f"{len(variable_groups.binary)} binarias, "
        f"{len(variable_groups.categorical)} categoricas"
    )
    atlp_summary = build_atlp_summary(
        clustered_df=clustered_df,
        cluster_summary=cluster_summary,
        variable_groups=variable_groups,
        invert_vars=set(args.invert_vars),
        low_thresh=args.low_thresh,
        high_thresh=args.high_thresh,
    )
    atlp_summary.to_csv(output_dir / "atlp_summary.csv", index=False)

    categorical_context = build_categorical_context_summary(atlp_summary)
    categorical_context.to_csv(output_dir / "categorical_context_summary.csv", index=False)

    log("Calculando validacion estadistica...")
    numeric_tests = build_numeric_association_table(
        clustered_df=clustered_df,
        variables=variable_groups.numeric + variable_groups.binary,
    )
    numeric_tests.to_csv(output_dir / "numeric_association_tests.csv", index=False)

    categorical_tests = build_categorical_association_table(
        clustered_df=clustered_df,
        variables=variable_groups.categorical,
    )
    categorical_tests.to_csv(output_dir / "categorical_association_tests.csv", index=False)

    validation_report = build_validation_report(
        profile_table=profile_table,
        atlp_summary=atlp_summary,
        numeric_tests=numeric_tests,
        categorical_tests=categorical_tests,
    )
    (output_dir / "profiling_validation_report.md").write_text(
        validation_report,
        encoding="utf-8",
    )

    if not args.skip_plots:
        log("Generando figuras...")
        try:
            plot_atlp_panel(atlp_summary, output_dir / "atlp_panel.png")
            plot_uncertainty_heatmap(atlp_summary, output_dir / "atlp_uncertainty_heatmap.png")
        except ModuleNotFoundError as exc:
            log(f"Se omiten las figuras porque falta una dependencia opcional: {exc.name}")

    log(f"Profiling ampliado generado en: {output_dir.resolve()}")
    log("Ficheros creados:")
    for name in [
        "cluster_profile_table.csv",
        "cluster_profile_report.txt",
        "atlp_summary.csv",
        "categorical_context_summary.csv",
        "numeric_association_tests.csv",
        "categorical_association_tests.csv",
        "profiling_validation_report.md",
        "atlp_panel.png",
        "atlp_uncertainty_heatmap.png",
    ]:
        if (output_dir / name).exists():
            log(f"- {name}")


if __name__ == "__main__":
    main()
