from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback
from uuid import uuid4

import sklearn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
IMPORT_DIRS = [
    BASE_DIR / "IDSS",
    BASE_DIR / "Source" / "LLM",
    BASE_DIR / "Source" / "Extraction",
]
for import_dir in IMPORT_DIRS:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from new_patient_pipeline import (
    DEFAULT_CLUSTER_PROFILE,
    DEFAULT_CLUSTERING_DIR,
    DEFAULT_KNOWLEDGE_BASE,
    DEFAULT_NEIGHBORS,
    DEFAULT_XGB_DIR,
    assign_cluster_to_patient,
    build_patient_feature_snapshot,
    build_prompt,
    compute_xgb_outputs,
    extract_patient_identifiers,
    load_clustering_artifacts,
    load_csv,
    load_json,
    load_single_patient,
)
from ollama_explainer import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_MODEL,
    build_augmented_prompt,
    call_ollama_validated,
    find_patient_rows,
)
from pdf_to_patient_csv import build_patient_csv_from_pdf


WEB_DIR = BASE_DIR / "Web"
UPLOAD_DIR = BASE_DIR / "outputs" / "web_uploads"
PROMPT_OUTPUT_DIR = BASE_DIR / "outputs" / "gemini_prompts"
LLM_OUTPUT_DIR = BASE_DIR / "outputs" / "ollama_responses"
PDF_TEXT_OUTPUT_DIR = BASE_DIR / "outputs" / "pdf_extraction"
HISTORY_PATH = BASE_DIR / "outputs" / "web_patient_history.json"
STATIC_MOUNT = "/static"
EXPECTED_SKLEARN_VERSION = "1.7.2"


app = FastAPI(title="Sepsis Mortality IDSS Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(STATIC_MOUNT, StaticFiles(directory=str(WEB_DIR)), name="static")


def ensure_supported_sklearn_version() -> None:
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            "Version incompatible de scikit-learn. "
            f"Se esperaba {EXPECTED_SKLEARN_VERSION} y se detecto {sklearn.__version__}. "
            "Reinstala dependencias con 'pip install -r requirements.txt' "
            "y 'pip install -r Web\\requirements.txt'."
        )


def build_case_context_from_csv(patient_input_path: Path) -> tuple[dict, str]:
    xgb_dir = Path(DEFAULT_XGB_DIR)
    clustering_dir = Path(DEFAULT_CLUSTERING_DIR)
    cluster_profile_path = Path(DEFAULT_CLUSTER_PROFILE)
    knowledge_base_path = Path(DEFAULT_KNOWLEDGE_BASE)

    patient_df = load_single_patient(patient_input_path)
    patient_row = patient_df.iloc[0]

    xgb_outputs, xgb_metadata = compute_xgb_outputs(
        patient_df=patient_df,
        xgb_dir=xgb_dir,
        top_n=5,
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
        n_neighbors=DEFAULT_NEIGHBORS,
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
    return case_context, prompt


def append_frontend_metadata(
    prompt: str,
    nombre: str,
    apellido: str,
    edad: int,
    altura: float,
    peso: float,
    genero: str,
    etnia: str,
    extraction_summary: dict | None = None,
) -> str:
    frontend_context = {
        "nombre": nombre,
        "apellido": apellido,
        "edad_formulario": edad,
        "altura_cm": altura,
        "peso_kg": peso,
        "genero_formulario": genero,
        "etnia_formulario": etnia,
    }
    frontend_json = json.dumps(frontend_context, ensure_ascii=False, indent=2)
    extraction_block = ""
    if extraction_summary:
        extraction_json = json.dumps(extraction_summary, ensure_ascii=False, indent=2)
        extraction_block = (
            "\n\nRESUMEN DE EXTRACCION DEL INFORME PDF\n"
            "======================================\n"
            "El CSV estructurado se ha generado automaticamente a partir del PDF. "
            "Ten en cuenta los campos detectados y los campos no encontrados en el informe.\n\n"
            f"{extraction_json}\n"
        )
    return (
        f"{prompt}\n\n"
        "METADATOS ADICIONALES DEL FORMULARIO WEB\n"
        "=======================================\n"
        "Usa estos datos solo como apoyo contextual. "
        "Si contradicen al CSV estructurado, prioriza el CSV.\n\n"
        f"{frontend_json}\n"
        f"{extraction_block}"
        "\nRECORDATORIO FINAL DE FORMATO\n"
        "=============================\n"
        "Devuelve solo el informe final para el medico. Maximo 180 palabras. "
        "No incluyas frases preferidas, frases a evitar ni notas de mantenimiento.\n"
    )


def persist_outputs(
    request_id: str,
    prompt: str,
    explanation: str,
) -> tuple[Path, Path]:
    PROMPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_path = PROMPT_OUTPUT_DIR / f"{request_id}_prompt.txt"
    explanation_path = LLM_OUTPUT_DIR / f"{request_id}_explanation.txt"

    prompt_path.write_text(prompt, encoding="utf-8")
    explanation_path.write_text(explanation, encoding="utf-8")
    return prompt_path, explanation_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return payload
    return []


def save_history(history: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize_history_entry(entry: dict) -> dict:
    keys = [
        "request_id",
        "paciente",
        "created_at",
        "updated_at",
        "predicted_probability",
        "predicted_risk_group",
        "cluster_label",
        "cluster",
        "comentario_medico",
    ]
    return {key: entry.get(key) for key in keys if key in entry}


def append_history_entry(entry: dict) -> None:
    history = load_history()
    history = [item for item in history if item.get("request_id") != entry["request_id"]]
    history.append(entry)
    save_history(history)


def find_history_entry(request_id: str) -> tuple[list[dict], dict]:
    history = load_history()
    for entry in history:
        if entry.get("request_id") == request_id:
            return history, entry
    raise HTTPException(status_code=404, detail="Paciente historico no encontrado.")


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/historial")
async def listar_historial():
    history = load_history()
    ordered = sorted(history, key=lambda item: item.get("created_at", ""), reverse=True)
    return {"status": "success", "items": [summarize_history_entry(item) for item in ordered]}


@app.get("/api/historial/{request_id}")
async def obtener_historial(request_id: str):
    _, entry = find_history_entry(request_id)
    return {"status": "success", "item": entry}


@app.patch("/api/historial/{request_id}/comentario")
async def actualizar_comentario_historial(
    request_id: str,
    comentario: str = Body("", embed=True),
):
    history, entry = find_history_entry(request_id)
    entry["comentario_medico"] = comentario.strip()
    entry["updated_at"] = utc_now_iso()
    save_history(history)
    return {"status": "success", "item": entry}


@app.post("/api/analizar")
async def analizar_paciente(
    nombre: str = Form(...),
    apellido: str = Form(...),
    edad: int = Form(...),
    altura: float = Form(...),
    peso: float = Form(...),
    genero: str = Form(...),
    etnia: str = Form(...),
    archivo: UploadFile = File(...),
):
    ensure_supported_sklearn_version()

    if not archivo.filename:
        raise HTTPException(status_code=400, detail="Debes subir un archivo CSV o PDF valido.")

    original_filename = archivo.filename.lower()
    is_csv = original_filename.endswith(".csv")
    is_pdf = original_filename.endswith(".pdf")
    if not (is_csv or is_pdf):
        raise HTTPException(status_code=400, detail="Debes subir un archivo CSV o PDF valido.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    request_id = f"patient_{uuid4().hex[:10]}"
    uploaded_input_path = UPLOAD_DIR / f"{request_id}{'.pdf' if is_pdf else '.csv'}"
    patient_csv_path = UPLOAD_DIR / f"{request_id}.csv"

    input_bytes = await archivo.read()
    uploaded_input_path.write_bytes(input_bytes)
    extraction_summary = None

    try:
        if is_pdf:
            extraction_summary = build_patient_csv_from_pdf(
                pdf_path=uploaded_input_path,
                output_csv_path=patient_csv_path,
                template_csv_path=BASE_DIR / "IDSS" / "new_patient.csv",
                form_context={
                    "edad": edad,
                    "genero": genero,
                    "etnia": etnia,
                },
                extracted_text_path=PDF_TEXT_OUTPUT_DIR / f"{request_id}_text.txt",
            )
        else:
            patient_csv_path = uploaded_input_path

        case_context, base_prompt = build_case_context_from_csv(patient_csv_path)

        database_path = Path(DEFAULT_DATABASE_PATH)
        matched_rows = []
        if database_path.exists():
            database_df = load_csv(database_path)
            matched_rows = find_patient_rows(
                database_df=database_df,
                case_context=case_context,
                max_rows=1,
            )

        prompt_with_db = build_augmented_prompt(
            base_prompt=base_prompt,
            matched_rows=matched_rows,
        )
        final_prompt = append_frontend_metadata(
            prompt=prompt_with_db,
            nombre=nombre,
            apellido=apellido,
            edad=edad,
            altura=altura,
            peso=peso,
            genero=genero,
            etnia=etnia,
            extraction_summary=extraction_summary,
        )
        explanation = call_ollama_validated(
            prompt_text=final_prompt,
            model=DEFAULT_MODEL,
            host=None,
            temperature=0.1,
        )
        prompt_path, explanation_path = persist_outputs(
            request_id=request_id,
            prompt=final_prompt,
            explanation=explanation,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando el paciente: {exc}") from exc

    model_outputs = case_context["model_outputs"]
    cluster_assignment = case_context["cluster_assignment"]
    created_at = utc_now_iso()
    history_entry = {
        "request_id": request_id,
        "created_at": created_at,
        "updated_at": created_at,
        "paciente": f"{nombre} {apellido}",
        "formulario": {
            "nombre": nombre,
            "apellido": apellido,
            "edad": edad,
            "altura": altura,
            "peso": peso,
            "genero": genero,
            "etnia": etnia,
        },
        "comentario_medico": "",
        "predicted_probability": model_outputs["predicted_probability"],
        "predicted_risk_group": model_outputs["predicted_risk_group"],
        "cluster_label": cluster_assignment["cluster_label"],
        "cluster": cluster_assignment["cluster"],
        "explanation": explanation,
        "prompt_path": str(prompt_path),
        "explanation_path": str(explanation_path),
        "input_type": "pdf" if is_pdf else "csv",
        "uploaded_input_path": str(uploaded_input_path),
        "generated_csv_path": str(patient_csv_path),
        "extraction_summary": extraction_summary,
        "case_context": case_context,
    }
    append_history_entry(history_entry)

    return {
        "status": "success",
        "paciente": f"{nombre} {apellido}",
        "request_id": request_id,
        "created_at": created_at,
        "predicted_probability": model_outputs["predicted_probability"],
        "predicted_risk_group": model_outputs["predicted_risk_group"],
        "cluster_label": cluster_assignment["cluster_label"],
        "cluster": cluster_assignment["cluster"],
        "comentario_medico": "",
        "explanation": explanation,
        "input_type": "pdf" if is_pdf else "csv",
        "generated_csv_path": str(patient_csv_path),
        "extraction_summary": extraction_summary,
        "prompt_path": str(prompt_path),
        "explanation_path": str(explanation_path),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
