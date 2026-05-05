from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Falta la dependencia 'xgboost'. Instalala en el entorno actual con "
        "'python -m pip install xgboost' o 'python -m pip install -r requirements.txt'."
    ) from exc
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


DEFAULT_INPUT = Path("bbdd") / "BaseDatos_imputada_knn.csv"
DEFAULT_OUTPUT = Path("outputs") / "xgboost_explainability"
DEFAULT_CLUSTERING_ASSIGNMENTS = (
    Path("outputs") / "clustering_clinical" / "best_patient_assignments.csv"
)
DEFAULT_CATEGORICAL_COLUMNS = [
    "gender",
    "ethnicity",
    "admission_type",
    "admission_location",
    "marital_status",
    "gcs_unable",
]
DEFAULT_ID_COLUMNS = [
    "row_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "patient_id",
]
POST_OUTCOME_COLUMNS = [
    "deathoffset",
    "unitdischargeoffset",
    "hospitaldischargeoffset",
    "los_icu",
    "los_hospital",
]


def log(message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena un modelo XGBoost para mortalidad en sepsis y genera "
            "salidas de explicabilidad listas para usar en el IDSS."
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
        help="Columna binaria objetivo a predecir.",
    )
    parser.add_argument(
        "--categorical-cols",
        nargs="*",
        default=DEFAULT_CATEGORICAL_COLUMNS,
        help="Columnas categoricas codificadas numericamente.",
    )
    parser.add_argument(
        "--id-cols",
        nargs="*",
        default=DEFAULT_ID_COLUMNS,
        help="Columnas identificadoras que no se usan como features.",
    )
    parser.add_argument(
        "--clustering-assignments",
        default=str(DEFAULT_CLUSTERING_ASSIGNMENTS),
        help=(
            "CSV opcional con la asignacion de clustering. "
            "Se usa solo como contexto interpretativo, no como feature."
        ),
    )
    parser.add_argument(
        "--disable-clustering-context",
        action="store_true",
        help="No mezcla informacion del clustering en las salidas.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-n-explanations", type=int, default=5)
    parser.add_argument("--risk-low-threshold", type=float, default=0.20)
    parser.add_argument("--risk-high-threshold", type=float, default=0.60)
    parser.add_argument("--classification-threshold", type=float, default=0.50)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-child-weight", type=float, default=3.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
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

    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def maybe_sample_rows(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state).sort_values("row_id").reset_index(drop=True)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str],
    id_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    excluded = {target_col, *id_cols, *POST_OUTCOME_COLUMNS}
    feature_candidates = [col for col in df.columns if col not in excluded]

    valid_feature_cols: list[str] = []
    for col in feature_candidates:
        if df[col].nunique(dropna=False) <= 1:
            continue
        valid_feature_cols.append(col)

    categorical = [col for col in categorical_cols if col in valid_feature_cols]
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


def simplify_feature_name(feature_name: str) -> str:
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "", 1)
    if feature_name.startswith("cat__"):
        return feature_name.replace("cat__", "", 1)
    return feature_name


def assign_risk_group(probability: float, low_threshold: float, high_threshold: float) -> str:
    if probability < low_threshold:
        return "low"
    if probability < high_threshold:
        return "medium"
    return "high"


def build_model(args: argparse.Namespace, scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )


def compute_scale_pos_weight(y: pd.Series) -> float:
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives == 0:
        return 1.0
    return negatives / positives


def compute_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "classification_report": classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def load_clustering_context(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    clustering_df = pd.read_csv(path)
    expected = {"row_id", "cluster", "cluster_label"}
    if not expected.issubset(clustering_df.columns):
        return None

    keep_cols = [col for col in ["row_id", "cluster", "cluster_label", "severity_rank"] if col in clustering_df.columns]
    return clustering_df[keep_cols].copy()


def format_top_features(
    contribution_row: np.ndarray,
    feature_names: list[str],
    top_n: int,
    positive: bool,
) -> str:
    if positive:
        candidate_idx = np.where(contribution_row > 0)[0]
        ordered_idx = candidate_idx[np.argsort(contribution_row[candidate_idx])[::-1]]
    else:
        candidate_idx = np.where(contribution_row < 0)[0]
        ordered_idx = candidate_idx[np.argsort(contribution_row[candidate_idx])]

    if len(ordered_idx) == 0:
        return "none"

    parts: list[str] = []
    for idx in ordered_idx[:top_n]:
        parts.append(f"{feature_names[idx]} ({contribution_row[idx]:+.4f})")
    return ", ".join(parts)


def build_global_explanations(
    contributions: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    mean_abs = np.abs(contributions).mean(axis=0)
    mean_signed = contributions.mean(axis=0)
    global_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_contribution": mean_abs,
            "mean_signed_contribution": mean_signed,
        }
    )
    return global_df.sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)


def build_patient_explanations(
    scored_df: pd.DataFrame,
    contributions: np.ndarray,
    feature_names: list[str],
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, (_, scored_row) in enumerate(scored_df.iterrows()):
        rows.append(
            {
                "row_id": int(scored_row["row_id"]),
                "predicted_probability": float(scored_row["predicted_probability"]),
                "predicted_risk_group": scored_row["predicted_risk_group"],
                "true_label": int(scored_row["true_label"]),
                "top_positive_features": format_top_features(
                    contributions[idx],
                    feature_names,
                    top_n=top_n,
                    positive=True,
                ),
                "top_negative_features": format_top_features(
                    contributions[idx],
                    feature_names,
                    top_n=top_n,
                    positive=False,
                ),
            }
        )
    explanations_df = pd.DataFrame(rows)

    extra_cols = [col for col in ["subject_id", "hadm_id", "stay_id", "cluster", "cluster_label", "severity_rank"] if col in scored_df.columns]
    if extra_cols:
        explanations_df = explanations_df.merge(
            scored_df[["row_id", *extra_cols]].drop_duplicates(),
            on="row_id",
            how="left",
        )
    return explanations_df


def build_cluster_explainability_summary(
    scored_df: pd.DataFrame,
    contributions: np.ndarray,
    feature_names: list[str],
    top_n: int,
) -> pd.DataFrame:
    if "cluster_label" not in scored_df.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    feature_matrix = np.abs(contributions)
    for cluster_label, group_idx in scored_df.groupby("cluster_label").groups.items():
        idx = np.array(list(group_idx))
        cluster_abs = feature_matrix[idx].mean(axis=0)
        top_idx = np.argsort(cluster_abs)[::-1][:top_n]
        rows.append(
            {
                "cluster_label": cluster_label,
                "cluster": scored_df.loc[idx[0], "cluster"] if "cluster" in scored_df.columns else np.nan,
                "severity_rank": scored_df.loc[idx[0], "severity_rank"] if "severity_rank" in scored_df.columns else np.nan,
                "n_patients_test": int(len(idx)),
                "observed_mortality_test": float(scored_df.loc[idx, "true_label"].mean()),
                "mean_predicted_probability": float(scored_df.loc[idx, "predicted_probability"].mean()),
                "top_explanatory_features": ", ".join(
                    f"{feature_names[feature_idx]} ({cluster_abs[feature_idx]:.4f})"
                    for feature_idx in top_idx
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["severity_rank", "mean_predicted_probability"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def build_llm_ready_summary(row: pd.Series) -> str:
    parts = [
        f"Predicted mortality risk: {row['predicted_probability']:.3f} ({row['predicted_risk_group']})."
    ]

    if "cluster_label" in row and pd.notna(row["cluster_label"]):
        cluster_text = f"Clinical severity cluster: {row['cluster_label']}"
        if "severity_rank" in row and pd.notna(row["severity_rank"]):
            cluster_text += f" (severity rank {int(row['severity_rank'])})"
        cluster_text += "."
        parts.append(cluster_text)

    parts.append(f"Main factors increasing risk: {row['top_positive_features']}.")
    parts.append(f"Main factors decreasing risk: {row['top_negative_features']}.")
    parts.append(
        "Use this together with the sepsis knowledge base to propose monitoring or treatment actions."
    )
    return " ".join(parts)


def build_llm_ready_output(patient_explanations: pd.DataFrame) -> pd.DataFrame:
    llm_df = patient_explanations.copy()
    llm_df["llm_summary"] = llm_df.apply(build_llm_ready_summary, axis=1)
    ordered_columns = [
        col
        for col in [
            "row_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "predicted_probability",
            "predicted_risk_group",
            "true_label",
            "cluster",
            "cluster_label",
            "severity_rank",
            "top_positive_features",
            "top_negative_features",
            "llm_summary",
        ]
        if col in llm_df.columns
    ]
    return llm_df[ordered_columns].copy()


def save_outputs(
    output_dir: Path,
    model: xgb.XGBClassifier,
    preprocessor: ColumnTransformer,
    metrics: dict[str, object],
    global_explanations: pd.DataFrame,
    scored_df: pd.DataFrame,
    patient_explanations: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    llm_ready_output: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    global_explanations.to_csv(output_dir / "global_feature_importance.csv", index=False)
    scored_df.to_csv(output_dir / "test_predictions.csv", index=False)
    patient_explanations.to_csv(output_dir / "patient_explanations.csv", index=False)
    llm_ready_output.to_csv(output_dir / "llm_ready_patient_context.csv", index=False)
    if not cluster_summary.empty:
        cluster_summary.to_csv(output_dir / "cluster_explainability_summary.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    model.save_model(str(output_dir / "xgboost_model.json"))
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    clustering_assignments_path = Path(args.clustering_assignments)

    log(f"[xgboost] Cargando datos desde: {input_path}")
    df = load_dataset(input_path)
    if args.max_rows is not None:
        df = maybe_sample_rows(df, max_rows=args.max_rows, random_state=args.random_state)
        log(f"[xgboost] Filas usadas tras muestreo: {len(df)}")
    log(f"[xgboost] Dataset listo: {len(df)} filas x {len(df.columns)} columnas")

    if args.target_col not in df.columns:
        raise RuntimeError(f"No existe la columna objetivo '{args.target_col}' en el dataset.")

    feature_cols, numeric_cols, categorical_cols = select_feature_columns(
        df=df,
        target_col=args.target_col,
        categorical_cols=args.categorical_cols,
        id_cols=args.id_cols,
    )
    log(
        f"[xgboost] Features del modelo: {len(feature_cols)} "
        f"({len(numeric_cols)} numericas, {len(categorical_cols)} categoricas)"
    )

    meta_cols = [col for col in args.id_cols if col in df.columns]
    meta_cols.append(args.target_col)
    meta_df = df[meta_cols].copy()

    X = df[feature_cols]
    y = df[args.target_col].astype(int)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        meta_df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    log("[xgboost] Preparando matrices de entrenamiento y test...")
    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)
    transformed_feature_names = [
        simplify_feature_name(name) for name in preprocessor.get_feature_names_out()
    ]
    log(
        f"[xgboost] Matrices preparadas: train={X_train_prepared.shape}, "
        f"test={X_test_prepared.shape}"
    )

    scale_pos_weight = compute_scale_pos_weight(y_train)
    log(f"[xgboost] scale_pos_weight calculado: {scale_pos_weight:.3f}")
    model = build_model(args=args, scale_pos_weight=scale_pos_weight)

    log("[xgboost] Entrenando modelo...")
    model.fit(X_train_prepared, y_train)

    log("[xgboost] Generando predicciones...")
    probabilities = model.predict_proba(X_test_prepared)[:, 1]
    predictions = (probabilities >= args.classification_threshold).astype(int)
    metrics = compute_metrics(y_true=y_test, probabilities=probabilities, predictions=predictions)

    scored_df = meta_test.reset_index(drop=True).copy()
    scored_df = scored_df.rename(columns={args.target_col: "true_label"})
    scored_df["predicted_probability"] = probabilities
    scored_df["predicted_label"] = predictions
    scored_df["predicted_risk_group"] = [
        assign_risk_group(prob, args.risk_low_threshold, args.risk_high_threshold)
        for prob in probabilities
    ]

    clustering_context = None
    if not args.disable_clustering_context:
        clustering_context = load_clustering_context(clustering_assignments_path)
        if clustering_context is not None:
            log("[xgboost] Mezclando contexto de clustering en las salidas...")
            scored_df = scored_df.merge(clustering_context, on="row_id", how="left")
        else:
            log("[xgboost] No se encontro un CSV valido de clustering; se continua sin esa capa.")

    log("[xgboost] Calculando explicabilidad con contribuciones nativas de XGBoost...")
    dtest = xgb.DMatrix(X_test_prepared)
    contribution_matrix = model.get_booster().predict(dtest, pred_contribs=True)
    base_values = contribution_matrix[:, -1]
    feature_contributions = contribution_matrix[:, :-1]
    scored_df["xgb_base_value"] = base_values

    global_explanations = build_global_explanations(
        contributions=feature_contributions,
        feature_names=transformed_feature_names,
    )
    patient_explanations = build_patient_explanations(
        scored_df=scored_df,
        contributions=feature_contributions,
        feature_names=transformed_feature_names,
        top_n=args.top_n_explanations,
    )
    llm_ready_output = build_llm_ready_output(patient_explanations=patient_explanations)
    cluster_summary = build_cluster_explainability_summary(
        scored_df=scored_df,
        contributions=feature_contributions,
        feature_names=transformed_feature_names,
        top_n=args.top_n_explanations,
    )

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir.resolve()),
        "rows": int(len(df)),
        "target_col": args.target_col,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "post_outcome_columns_excluded": POST_OUTCOME_COLUMNS,
        "test_size": args.test_size,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "scale_pos_weight": scale_pos_weight,
        "classification_threshold": args.classification_threshold,
        "risk_thresholds": {
            "low": args.risk_low_threshold,
            "high": args.risk_high_threshold,
        },
        "clustering_context_path": (
            str(clustering_assignments_path) if clustering_context is not None else None
        ),
        "clustering_used_as_feature": False,
        "transformed_feature_count": len(transformed_feature_names),
    }

    log(f"[xgboost] Guardando resultados en: {output_dir}")
    save_outputs(
        output_dir=output_dir,
        model=model,
        preprocessor=preprocessor,
        metrics=metrics,
        global_explanations=global_explanations,
        scored_df=scored_df,
        patient_explanations=patient_explanations,
        cluster_summary=cluster_summary,
        llm_ready_output=llm_ready_output,
        metadata=metadata,
    )

    log("[xgboost] Entrenamiento y explicabilidad completados.")
    log(f"[xgboost] ROC AUC: {metrics['roc_auc']:.4f}")
    log(f"[xgboost] Average Precision: {metrics['average_precision']:.4f}")
    log(f"[xgboost] Brier Score: {metrics['brier_score']:.4f}")


if __name__ == "__main__":
    main()
