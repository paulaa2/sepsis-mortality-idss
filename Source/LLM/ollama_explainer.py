from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ollama import Client
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Falta la dependencia 'ollama'. Instalala con "
        "'python -m pip install ollama' o 'python -m pip install -r requirements.txt'."
    ) from exc


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    return current.parent


REPO_ROOT = find_repo_root()
DEFAULT_PROMPT_PATH = REPO_ROOT / "outputs" / "gemini_prompts" / "new_patient_prompt.txt"
DEFAULT_DATABASE_PATH = (
    REPO_ROOT / "outputs" / "xgboost_explainability" / "llm_ready_patient_context.csv"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "ollama_responses" / "new_patient_explanation.txt"
DEFAULT_MODEL = "medgemma:4b"
FINAL_OUTPUT_CONTRACT = """

INSTRUCCION FINAL DE SALIDA
===========================
Devuelve solo el informe clinico final. No expliques el prompt, no incluyas
frases preferidas, frases a evitar, notas de mantenimiento ni resumenes largos.
Maximo 180 palabras. Usa exactamente estos cuatro encabezados:
RIESGO:
LECTURA CLINICA:
CONDUCTA RECOMENDADA:
VIGILAR / COMPLETAR:
"""
STRICT_LANGUAGE_CONTRACT = """

VALIDACION DE IDIOMA Y FORMATO
==============================
Responde exclusivamente en espanol. Usa solo caracteres latinos habituales
del espanol y no mezcles otros alfabetos. Si tienes dudas, escribe una frase
breve y conservadora en espanol.
"""
REQUIRED_RESPONSE_HEADERS = [
    "RIESGO:",
    "LECTURA CLINICA:",
    "CONDUCTA RECOMENDADA:",
    "VIGILAR / COMPLETAR:",
]
ALLOWED_NON_ASCII = set("áéíóúÁÉÍÓÚñÑüÜçÇàèòÀÈÒ¿¡ºª")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Carga un prompt generado por new_patient_pipeline.py, "
            "anade contexto relevante desde la base de datos y pide a Ollama "
            "una explicacion clinica en lenguaje natural."
        )
    )
    parser.add_argument(
        "--prompt-path",
        default=str(DEFAULT_PROMPT_PATH),
        help="Ruta al prompt final generado previamente.",
    )
    parser.add_argument(
        "--database-path",
        default=str(DEFAULT_DATABASE_PATH),
        help="CSV con el contexto historico preparado para LLM.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Modelo local disponible en Ollama.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host de Ollama, por ejemplo http://localhost:11434.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Ruta donde guardar la respuesta generada por Ollama.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura de generacion para Ollama.",
    )
    parser.add_argument(
        "--max-context-rows",
        type=int,
        default=1,
        help="Numero maximo de filas historicas a adjuntar si hay coincidencias.",
    )
    return parser


def sniff_csv_format(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()

    if header.count(";") > header.count(","):
        return ";", ","
    return ",", "."


def load_csv(path: Path) -> pd.DataFrame:
    sep, decimal = sniff_csv_format(path)
    return pd.read_csv(path, sep=sep, decimal=decimal)


def extract_case_context(prompt_text: str) -> dict[str, Any]:
    marker = "CONTEXTO DEL CASO"
    if marker not in prompt_text:
        return {}

    section = prompt_text.split(marker, maxsplit=1)[1]
    json_start = section.find("{")
    if json_start == -1:
        return {}

    json_text = section[json_start:].strip()
    return json.loads(json_text)


def find_patient_rows(
    database_df: pd.DataFrame,
    case_context: dict[str, Any],
    max_rows: int,
) -> list[dict[str, Any]]:
    identifiers = case_context.get("patient_identifiers", {})
    candidate_columns = ["row_id", "subject_id", "hadm_id", "stay_id", "patient_id"]

    filtered = database_df.copy()
    any_filter = False
    for column in candidate_columns:
        if column not in filtered.columns or column not in identifiers:
            continue
        filtered = filtered[filtered[column].astype(str) == str(identifiers[column])]
        any_filter = True

    if not any_filter or filtered.empty:
        return []

    selected_columns = [
        column
        for column in [
            "row_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "predicted_probability",
            "predicted_risk_group",
            "cluster_label",
            "severity_rank",
            "top_positive_features",
            "top_negative_features",
            "llm_summary",
        ]
        if column in filtered.columns
    ]

    rows = filtered[selected_columns].head(max_rows)
    return rows.where(pd.notna(rows), None).to_dict(orient="records")


def build_augmented_prompt(
    base_prompt: str,
    matched_rows: list[dict[str, Any]],
) -> str:
    if not matched_rows:
        return (
            f"{base_prompt}\n\n"
            "CONTEXTO ADICIONAL DESDE LA BASE DE DATOS\n"
            "=========================================\n"
            "No se encontro una fila historica coincidente en el CSV de contexto.\n"
            f"{FINAL_OUTPUT_CONTRACT}\n"
        )

    db_context_json = json.dumps(matched_rows, ensure_ascii=False, indent=2)
    return (
        f"{base_prompt}\n\n"
        "CONTEXTO ADICIONAL DESDE LA BASE DE DATOS\n"
        "=========================================\n"
        "Usa esta informacion historica solo como apoyo interpretativo adicional.\n"
        "No sustituyas el contexto principal del caso por esta fila.\n\n"
        f"{db_context_json}\n"
        f"{FINAL_OUTPUT_CONTRACT}\n"
    )


def normalize_for_validation(text: str) -> str:
    return (
        text.upper()
        .replace("CLÍNICA", "CLINICA")
        .replace("SEGÚN", "SEGUN")
    )


def has_required_headers(text: str) -> bool:
    normalized = normalize_for_validation(text)
    return all(header in normalized for header in REQUIRED_RESPONSE_HEADERS)


def suspicious_character_ratio(text: str) -> float:
    if not text:
        return 1.0

    suspicious = 0
    checked = 0
    for char in text:
        if char.isspace() or char in "\n\r\t":
            continue
        checked += 1
        if ord(char) <= 127 or char in ALLOWED_NON_ASCII:
            continue
        suspicious += 1

    if checked == 0:
        return 1.0
    return suspicious / checked


def looks_like_spanish_clinical_response(text: str) -> bool:
    normalized = normalize_for_validation(text)
    spanish_markers = [
        "PACIENTE",
        "RIESGO",
        "CLINICA",
        "VALORAR",
        "VIGILAR",
        "REVISAR",
        "MORTALIDAD",
        "FENOTIPO",
    ]
    return sum(marker in normalized for marker in spanish_markers) >= 3


def is_valid_llm_response(text: str) -> bool:
    clean = text.strip()
    if len(clean) < 40:
        return False
    if not has_required_headers(clean):
        return False
    if suspicious_character_ratio(clean) > 0.03:
        return False
    if not looks_like_spanish_clinical_response(clean):
        return False
    return True


def fallback_llm_response() -> str:
    return (
        "RIESGO:\n"
        "No se ha podido generar una lectura clinica fiable automaticamente.\n\n"
        "LECTURA CLINICA:\n"
        "La respuesta del modelo de lenguaje no ha superado la validacion de idioma "
        "o formato, por lo que debe descartarse.\n\n"
        "CONDUCTA RECOMENDADA:\n"
        "- Hacer ahora: revisar manualmente la probabilidad, el grupo de riesgo y el fenotipo.\n"
        "- Valorar segun clinica: completar datos criticos del paciente.\n"
        "- No indicado de rutina: usar una respuesta generativa no validada.\n\n"
        "VIGILAR / COMPLETAR:\n"
        "Validar el caso con criterio clinico."
    )


def call_ollama(
    prompt_text: str,
    model: str,
    host: str | None,
    temperature: float,
) -> str:
    client = Client(host=host) if host else Client()
    response = client.generate(
        model=model,
        prompt=prompt_text,
        options={"temperature": temperature},
    )
    return response["response"].strip()


def call_ollama_validated(
    prompt_text: str,
    model: str,
    host: str | None,
    temperature: float,
    max_retries: int = 2,
) -> str:
    prompt_with_contract = f"{prompt_text}\n{STRICT_LANGUAGE_CONTRACT}"
    retry_instruction = (
        "\n\nLa respuesta anterior no era valida. Reescribe desde cero, solo en espanol, "
        "con exactamente estos encabezados: RIESGO:, LECTURA CLINICA:, "
        "CONDUCTA RECOMENDADA:, VIGILAR / COMPLETAR:. No uses otros idiomas."
    )

    for attempt in range(max_retries + 1):
        current_prompt = prompt_with_contract if attempt == 0 else prompt_with_contract + retry_instruction
        response = call_ollama(
            prompt_text=current_prompt,
            model=model,
            host=host,
            temperature=temperature,
        )
        if is_valid_llm_response(response):
            return response

    return fallback_llm_response()


def main() -> None:
    args = build_parser().parse_args()

    prompt_path = Path(args.prompt_path)
    database_path = Path(args.database_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_text = prompt_path.read_text(encoding="utf-8")
    case_context = extract_case_context(prompt_text)

    matched_rows: list[dict[str, Any]] = []
    if database_path.exists():
        database_df = load_csv(database_path)
        matched_rows = find_patient_rows(
            database_df=database_df,
            case_context=case_context,
            max_rows=args.max_context_rows,
        )

    final_prompt = build_augmented_prompt(
        base_prompt=prompt_text,
        matched_rows=matched_rows,
    )
    explanation = call_ollama_validated(
        prompt_text=final_prompt,
        model=args.model,
        host=args.host,
        temperature=min(args.temperature, 0.1),
    )

    output_path.write_text(explanation, encoding="utf-8")

    print(f"Prompt usado: {prompt_path.resolve()}")
    print(f"CSV de contexto: {database_path.resolve()}")
    print(f"Modelo Ollama: {args.model}")
    print(f"Respuesta guardada en: {output_path.resolve()}")


if __name__ == "__main__":
    main()
