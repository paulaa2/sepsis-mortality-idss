from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CLUSTERING_DIR = Path("outputs") / "clustering_clinical"
DEFAULT_XGB_DIR = Path("outputs") / "xgboost_explainability"


RENAL_FEATURES = {"creatinine_max", "creatinine_min", "creatinine_score", "bun_max", "bun_min", "bun_score", "urineoutput", "uo_score", "sodium_min", "sodium_max", "glucose_max", "glucose_min"}
NEURO_FEATURES = {"gcs_score", "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable"}
INFLAMMATORY_FEATURES = {"temp_score", "temperature_max", "temperature_min", "wbc_score", "wbc_max", "wbc_min", "resp_rate_score", "resp_rate_max", "resp_rate_min", "heart_rate_max", "heart_rate_min", "hr_score"}
HEMODYNAMIC_FEATURES = {"mbp_score", "mbp_min", "mbp_max"}
SEVERITY_FEATURES = {"apsiii", "apsiii_prob", "sepsis3", "admission_age"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera perfiles de clustering reutilizables para el IDSS a partir "
            "de las salidas del clustering y, opcionalmente, del XGBoost."
        )
    )
    parser.add_argument(
        "--clustering-dir",
        default=str(DEFAULT_CLUSTERING_DIR),
        help="Carpeta con best_cluster_summary.csv y best_cluster_profiles.csv.",
    )
    parser.add_argument(
        "--xgb-dir",
        default=str(DEFAULT_XGB_DIR),
        help="Carpeta opcional con cluster_explainability_summary.csv del modelo.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta de salida. Por defecto usa <clustering-dir>/profiling.",
    )
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=5,
        help="Numero de variables altas y bajas a resumir por cluster.",
    )
    return parser


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero requerido: {path}")
    return pd.read_csv(path)


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
) -> str:
    votes = feature_family_votes(high_features)
    low_feature_set = set(low_features)

    if severity_rank is not None and pd.notna(severity_rank):
        severity_rank = int(severity_rank)
    else:
        severity_rank = None

    if severity_rank is not None and severity_rank >= 4:
        if {"apsiii", "creatinine_max", "bun_max", "sepsis3"}.intersection(low_feature_set):
            return "stable_low_severity"

    if votes["renal_metabolic"] >= 2 and votes["severity"] >= 1:
        return "renal_metabolic_high_severity"
    if votes["neurologic"] >= 2 and votes["inflammatory"] >= 1:
        return "neurologic_inflammatory"
    if votes["frailty"] >= 2 and votes["severity"] >= 1:
        return "older_frail"
    if votes["hemodynamic"] >= 2 and votes["severity"] >= 1:
        return "hemodynamic_instability"
    if votes["severity"] >= 2:
        return "moderate_to_high_severity"
    if severity_rank is not None and severity_rank >= 4:
        return "stable_low_severity"
    return "mixed_intermediate"


def extract_mean_value(row: pd.Series, feature: str) -> float | None:
    column_name = f"mean_{feature}"
    if column_name not in row.index or pd.isna(row[column_name]):
        return None
    return float(row[column_name])


def build_clinical_snapshot(row: pd.Series) -> str:
    pieces: list[str] = []
    for feature in ["apsiii", "admission_age", "creatinine_max", "bun_max", "mbp_min", "temperature_max", "urineoutput", "gcs_score"]:
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
        lines.append(f"{row.cluster_label} | {row.phenotype_name}")
        lines.append("-" * max(12, len(f"{row.cluster_label} | {row.phenotype_name}")))
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


def main() -> None:
    args = build_parser().parse_args()

    clustering_dir = Path(args.clustering_dir)
    xgb_dir = Path(args.xgb_dir)
    output_dir = Path(args.output_dir) if args.output_dir else clustering_dir / "profiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_summary = load_csv(clustering_dir / "best_cluster_summary.csv")
    cluster_profiles = load_csv(clustering_dir / "best_cluster_profiles.csv")
    xgb_cluster_summary = load_xgb_cluster_summary(xgb_dir)

    profiling_df = build_profile_table(
        cluster_summary=cluster_summary,
        cluster_profiles=cluster_profiles,
        xgb_cluster_summary=xgb_cluster_summary,
        top_n=args.top_n_features,
    )
    profiling_df.to_csv(output_dir / "cluster_profile_table.csv", index=False)

    report = build_report(profiling_df)
    (output_dir / "cluster_profile_report.txt").write_text(report, encoding="utf-8")

    print(f"Profiling generado en: {output_dir.resolve()}")
    print("Ficheros creados:")
    print("- cluster_profile_table.csv")
    print("- cluster_profile_report.txt")


if __name__ == "__main__":
    main()
