from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import joblib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Falta la dependencia 'joblib'. Instalala con "
        "'python -m pip install joblib' o 'python -m pip install -r requirements.txt'."
    ) from exc

def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    return current.parent


REPO_ROOT = find_repo_root()
TRAINING_SOURCE_DIR = REPO_ROOT / "Source" / "Training"
if str(TRAINING_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_SOURCE_DIR))

from XGBoost import assign_risk_group, format_top_features, simplify_feature_name


DEFAULT_PATIENT_INPUT = REPO_ROOT / "IDSS" / "new_patient.csv"
DEFAULT_XGB_DIR = REPO_ROOT / "outputs" / "xgboost_explainability"
DEFAULT_CLUSTERING_DIR = REPO_ROOT / "outputs" / "clustering_clinical"
DEFAULT_CLUSTER_PROFILE = DEFAULT_CLUSTERING_DIR / "profiling" / "cluster_profile_table.csv"
DEFAULT_KNOWLEDGE_BASE = REPO_ROOT / "KnowledgeSources" / "knowledge_base.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "gemini_prompts"
DEFAULT_NEIGHBORS = 15


PROMPT_HEADER = """Eres un asistente clinico para un IDSS de sepsis en UCI.

Tu salida la leera un medico. Debe ser breve, accionable y facil de escanear.

Usa EXCLUSIVAMENTE:
1. La base de conocimiento proporcionada.
2. El contexto estructurado del caso.

Reglas estrictas de estilo:
- Responde siempre en espanol.
- Maximo 180 palabras.
- No incluyas introducciones tipo "aqui tienes".
- No incluyas "frases preferidas", "frases a evitar", notas de mantenimiento,
  ni explicaciones sobre como has generado la respuesta.
- No repitas listas largas de variables del modelo. Cita como maximo 3 factores.
- No digas que el diagnostico esta confirmado.
- Si el riesgo es bajo, no propongas tratamiento agresivo de forma rutinaria.
- Si recomiendas antibioticos, fluidos, vasopresores o escalada, especifica
  claramente si es "hacer ahora", "valorar segun clinica" o "no indicado de rutina".
- Si faltan datos clave como lactato, cultivos, foco infeccioso, PAM actual o
  diuresis horaria, pide solo los mas relevantes.

Formato obligatorio:
RIESGO:
Una frase con probabilidad, grupo de riesgo y fenotipo.

LECTURA CLINICA:
Dos frases maximo. Explica que significa el patron, sin diagnosticar con certeza.

CONDUCTA RECOMENDADA:
- Hacer ahora: ...
- Valorar segun clinica: ...
- No indicado de rutina: ...

VIGILAR / COMPLETAR:
Maximo 3 items.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dado un paciente nuevo, genera la prediccion de mortalidad, "
            "lo asigna al cluster correspondiente y construye el prompt final para Gemini."
        )
    )
    parser.add_argument(
        "--patient-input",
        default=str(DEFAULT_PATIENT_INPUT),
        help="CSV con un unico paciente nuevo.",
    )
    parser.add_argument(
        "--xgb-dir",
        default=str(DEFAULT_XGB_DIR),
        help="Directorio con xgboost_model.json, preprocessor.joblib y run_metadata.json.",
    )
    parser.add_argument(
        "--clustering-dir",
        default=str(DEFAULT_CLUSTERING_DIR),
        help="Directorio con run_metadata.json del clustering.",
    )
    parser.add_argument(
        "--cluster-profile",
        default=str(DEFAULT_CLUSTER_PROFILE),
        help="CSV con el significado clinico de los clusters.",
    )
    parser.add_argument(
        "--knowledge-base",
        default=str(DEFAULT_KNOWLEDGE_BASE),
        help="Fichero de base de conocimiento.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directorio donde se guarda el prompt final.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_NEIGHBORS,
        help="Numero de pacientes historicos mas parecidos a usar para asignar el cluster.",
    )
    parser.add_argument(
        "--top-n-explanations",
        type=int,
        default=5,
        help="Numero de variables explicativas positivas/negativas a incluir.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Parametro reservado; no se usa si el clustering ya tiene artefactos persistidos.",
    )
    return parser


def sniff_csv_format(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()

    if header.count(";") > header.count(","):
        return ";", ","
    return ",", "."


def load_csv(path: Path, add_row_id: bool = False) -> pd.DataFrame:
    sep, decimal = sniff_csv_format(path)
    df = pd.read_csv(path, sep=sep, decimal=decimal)
    blank_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if blank_cols:
        df = df.rename(columns={blank_cols[0]: "Unnamed: 0"})
    if add_row_id and "row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_id"] = range(len(df))
    return df


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    prepared = df.copy()
    for column in columns:
        if column not in prepared.columns:
            prepared[column] = np.nan
    return prepared


def load_single_patient(path: Path) -> pd.DataFrame:
    df = load_csv(path, add_row_id=False)
    if len(df) != 1:
        raise ValueError(
            f"El fichero {path} debe contener exactamente un paciente, pero tiene {len(df)} filas."
        )
    return df.reset_index(drop=True)


def compute_xgb_outputs(
    patient_df: pd.DataFrame,
    xgb_dir: Path,
    top_n: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = load_json(xgb_dir / "run_metadata.json")
    feature_columns = list(metadata["feature_columns"])
    risk_low = float(metadata["risk_thresholds"]["low"])
    risk_high = float(metadata["risk_thresholds"]["high"])

    preprocessor = joblib.load(xgb_dir / "preprocessor.joblib")
    model = xgb.XGBClassifier()
    model.load_model(str(xgb_dir / "xgboost_model.json"))

    patient_features = ensure_columns(patient_df, feature_columns)[feature_columns].copy()
    X_prepared = preprocessor.transform(patient_features)

    probability = float(model.predict_proba(X_prepared)[:, 1][0])
    risk_group = assign_risk_group(probability, low_threshold=risk_low, high_threshold=risk_high)

    transformed_feature_names = [
        simplify_feature_name(name) for name in preprocessor.get_feature_names_out()
    ]
    dmatrix = xgb.DMatrix(X_prepared)
    contribution_matrix = model.get_booster().predict(dmatrix, pred_contribs=True)
    feature_contributions = contribution_matrix[0, :-1]

    outputs = {
        "predicted_probability": probability,
        "predicted_risk_group": risk_group,
        "top_positive_features": format_top_features(
            feature_contributions,
            transformed_feature_names,
            top_n=top_n,
            positive=True,
        ),
        "top_negative_features": format_top_features(
            feature_contributions,
            transformed_feature_names,
            top_n=top_n,
            positive=False,
        ),
    }
    return outputs, metadata


def load_clustering_artifacts(clustering_dir: Path, clustering_metadata: dict[str, Any]) -> tuple[Any, Any, Any, pd.DataFrame]:
    artifacts = clustering_metadata.get("artifacts", {})
    if not artifacts:
        raise FileNotFoundError(
            "No se encontraron artefactos persistidos del clustering. "
            "Ejecuta icu_sepsis_clustering.py una vez para guardarlos y luego reutilizarlos en inferencia."
        )

    preprocessor_path = clustering_dir / artifacts["preprocessor"]
    svd_path = clustering_dir / artifacts["svd"]
    scaler_path = clustering_dir / artifacts["scaler"]
    reference_embeddings_path = clustering_dir / artifacts["reference_embeddings"]

    missing = [
        path
        for path in [preprocessor_path, svd_path, scaler_path, reference_embeddings_path]
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Faltan artefactos del clustering. "
            f"Archivos ausentes: {missing_text}. "
            "Vuelve a ejecutar icu_sepsis_clustering.py para generarlos."
        )

    preprocessor = joblib.load(preprocessor_path)
    svd = joblib.load(svd_path)
    scaler = joblib.load(scaler_path)
    reference_embeddings = load_csv(reference_embeddings_path)
    return preprocessor, svd, scaler, reference_embeddings


def assign_cluster_to_patient(
    patient_df: pd.DataFrame,
    preprocessor: Any,
    svd: Any,
    scaler: Any,
    clustering_metadata: dict[str, Any],
    reference_embeddings_df: pd.DataFrame,
    cluster_profile_df: pd.DataFrame | None,
    n_neighbors: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    feature_columns = list(clustering_metadata["feature_columns"])
    patient_features = ensure_columns(patient_df, feature_columns)[feature_columns].copy()

    X_prepared = preprocessor.transform(patient_features)
    X_reduced = svd.transform(X_prepared) if svd is not None else X_prepared
    X_scaled = scaler.transform(X_reduced)

    embedding_columns = [col for col in reference_embeddings_df.columns if col.startswith("dim_")]
    if not embedding_columns:
        raise ValueError("El fichero de embeddings de referencia no contiene columnas dim_*.") 

    reference_matrix = reference_embeddings_df[embedding_columns].to_numpy(dtype=float)
    patient_vector = X_scaled[0].astype(float)

    distances = np.linalg.norm(reference_matrix - patient_vector, axis=1)
    order = np.argsort(distances)
    k = max(1, min(int(n_neighbors), len(order)))
    neighbor_idx = order[:k]
    neighbor_rows = reference_embeddings_df.iloc[neighbor_idx].copy()
    neighbor_distances = distances[neighbor_idx]
    weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
    neighbor_rows["similarity_weight"] = weights

    cluster_weights = (
        neighbor_rows.groupby(["cluster", "cluster_label"], as_index=False)["similarity_weight"].sum()
    )
    winner = cluster_weights.sort_values("similarity_weight", ascending=False).iloc[0]
    cluster = int(winner["cluster"])
    cluster_label = f"cluster_{cluster}"

    assignment = {
        "cluster": cluster,
        "cluster_label": cluster_label,
        "assignment_method": "similarity_to_historical_clustered_patients",
        "neighbors_used": k,
        "neighbor_cluster_vote": [
            {
                "cluster": int(row["cluster"]),
                "cluster_label": str(row["cluster_label"]),
                "weight": float(row["similarity_weight"]),
            }
            for _, row in cluster_weights.sort_values("similarity_weight", ascending=False).iterrows()
        ],
    }

    cluster_profile = None
    if cluster_profile_df is not None and not cluster_profile_df.empty:
        matches = cluster_profile_df[cluster_profile_df["cluster_label"] == cluster_label]
        if not matches.empty:
            row = matches.iloc[0]
            assignment["severity_rank"] = to_python_scalar(row.get("severity_rank"))
            keep_columns = [
                "cluster_label",
                "phenotype_name",
                "severity_rank",
                "top_high_features",
                "top_low_features",
                "clinical_snapshot",
                "xgb_observed_mortality_test",
                "xgb_mean_predicted_probability",
            ]
            cluster_profile = {
                column: to_python_scalar(row[column])
                for column in keep_columns
                if column in row.index and not pd.isna(row[column])
            }

    return assignment, cluster_profile


def extract_patient_identifiers(patient_row: pd.Series) -> dict[str, Any]:
    identifiers: dict[str, Any] = {}
    for column in ["row_id", "subject_id", "hadm_id", "stay_id", "patient_id"]:
        if column in patient_row.index and not pd.isna(patient_row[column]):
            identifiers[column] = to_python_scalar(patient_row[column])
    return identifiers


def build_patient_feature_snapshot(patient_row: pd.Series, xgb_metadata: dict[str, Any]) -> dict[str, Any]:
    feature_columns = list(xgb_metadata["feature_columns"])
    snapshot: dict[str, Any] = {}
    for column in feature_columns:
        if column not in patient_row.index:
            continue
        value = to_python_scalar(patient_row[column])
        if value is None:
            continue
        snapshot[column] = value
    return snapshot


def build_prompt(knowledge_base_text: str, case_context: dict[str, Any]) -> str:
    case_context_json = json.dumps(case_context, ensure_ascii=False, indent=2)
    return (
        f"{PROMPT_HEADER}\n\n"
        f"BASE DE CONOCIMIENTO\n"
        f"====================\n"
        f"{knowledge_base_text}\n\n"
        f"CONTEXTO DEL CASO\n"
        f"=================\n"
        f"{case_context_json}\n"
    )


def main() -> None:
    args = build_parser().parse_args()

    patient_input_path = Path(args.patient_input)
    xgb_dir = Path(args.xgb_dir)
    clustering_dir = Path(args.clustering_dir)
    cluster_profile_path = Path(args.cluster_profile)
    knowledge_base_path = Path(args.knowledge_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_df = load_single_patient(patient_input_path)
    patient_row = patient_df.iloc[0]

    xgb_outputs, xgb_metadata = compute_xgb_outputs(
        patient_df=patient_df,
        xgb_dir=xgb_dir,
        top_n=args.top_n_explanations,
    )

    clustering_metadata = load_json(clustering_dir / "run_metadata.json")
    cluster_profile_df = load_csv(cluster_profile_path) if cluster_profile_path.exists() else None

    cluster_preprocessor, svd, scaler, reference_embeddings_df = load_clustering_artifacts(
        clustering_dir=clustering_dir,
        clustering_metadata=clustering_metadata,
    )
    cluster_assignment, cluster_profile = assign_cluster_to_patient(
        patient_df=patient_df,
        preprocessor=cluster_preprocessor,
        svd=svd,
        scaler=scaler,
        clustering_metadata=clustering_metadata,
        reference_embeddings_df=reference_embeddings_df,
        cluster_profile_df=cluster_profile_df,
        n_neighbors=args.n_neighbors,
    )

    case_context = {
        "patient_identifiers": extract_patient_identifiers(patient_row),
        "model_outputs": xgb_outputs,
        "cluster_assignment": cluster_assignment,
        "cluster_profile": cluster_profile,
        "patient_features": build_patient_feature_snapshot(patient_row, xgb_metadata),
    }

    knowledge_base_text = knowledge_base_path.read_text(encoding="utf-8")
    prompt = build_prompt(knowledge_base_text=knowledge_base_text, case_context=case_context)

    patient_name = patient_input_path.stem
    prompt_path = output_dir / f"{patient_name}_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    print(f"Prompt guardado en: {prompt_path.resolve()}")
    print(f"Probabilidad predicha: {xgb_outputs['predicted_probability']:.4f}")
    print(f"Grupo de riesgo: {xgb_outputs['predicted_risk_group']}")
    print(f"Cluster asignado: {cluster_assignment['cluster_label']}")


if __name__ == "__main__":
    main()
