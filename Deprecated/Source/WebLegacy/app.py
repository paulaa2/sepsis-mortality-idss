from __future__ import annotations

import csv
import io
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


WEB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEB_DIR.parent
RUNTIME_DIR = WEB_DIR / "runtime"
INPUT_DIR = RUNTIME_DIR / "inputs"
LLM_OUTPUT_DIR = RUNTIME_DIR / "llm_outputs"
PROMPT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "gemini_prompts"
DEFAULT_LLM_MODEL = "medgemma:4b"

for directory in (INPUT_DIR, LLM_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PAID Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_uploaded_csv(csv_bytes: bytes) -> dict[str, str]:
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="El archivo CSV debe estar codificado en UTF-8.",
        ) from exc

    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="El CSV subido no contiene filas de datos.")

    if len(rows) != 1:
        raise HTTPException(
            status_code=400,
            detail="El CSV subido debe contener exactamente una fila para el nuevo paciente.",
        )

    return {str(key).strip(): "" if value is None else str(value).strip() for key, value in rows[0].items()}


def normalize_gender(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "masculino": "0",
        "hombre": "0",
        "male": "0",
        "m": "0",
        "femenino": "1",
        "mujer": "1",
        "female": "1",
        "f": "1",
        "otro": "2",
        "other": "2",
    }
    return mapping.get(normalized, value.strip())


def merge_patient_context(
    uploaded_row: dict[str, str],
    *,
    nombre: str,
    apellido: str,
    edad: int,
    altura: float,
    peso: float,
    genero: str,
    etnia: str,
) -> dict[str, str]:
    merged = dict(uploaded_row)

    merged["nombre"] = nombre.strip()
    merged["apellido"] = apellido.strip()
    merged["nombre_completo"] = f"{nombre.strip()} {apellido.strip()}".strip()
    merged["edad_formulario"] = str(edad)
    merged["altura_cm"] = str(altura)
    merged["peso_kg"] = str(peso)
    merged["genero_formulario"] = genero.strip()
    merged["etnia_formulario"] = etnia.strip()

    if not merged.get("admission_age"):
        merged["admission_age"] = str(edad)
    if not merged.get("gender"):
        merged["gender"] = normalize_gender(genero)
    if not merged.get("ethnicity"):
        merged["ethnicity"] = etnia.strip()

    for identifier in ("subject_id", "hadm_id", "stay_id", "patient_id", "row_id"):
        if not merged.get(identifier):
            merged[identifier] = str(uuid.uuid4().int % 10**8)

    return merged


def save_combined_csv(row: dict[str, str], filename_stub: str) -> Path:
    fieldnames = list(row.keys())
    output_path = INPUT_DIR / f"{filename_stub}.csv"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    return output_path


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "Sin detalle adicional."
        raise HTTPException(
            status_code=500,
            detail=f"Fallo ejecutando {' '.join(command)}: {stderr}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="No se pudo lanzar Python para ejecutar el pipeline.",
        ) from exc


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
    csv_bytes = await archivo.read()
    uploaded_row = parse_uploaded_csv(csv_bytes)

    merged_row = merge_patient_context(
        uploaded_row,
        nombre=nombre,
        apellido=apellido,
        edad=edad,
        altura=altura,
        peso=peso,
        genero=genero,
        etnia=etnia,
    )

    request_id = uuid.uuid4().hex[:12]
    patient_slug = f"patient_{request_id}"
    patient_csv_path = save_combined_csv(merged_row, patient_slug)
    prompt_path = PROMPT_OUTPUT_DIR / f"{patient_slug}_prompt.txt"
    llm_output_path = LLM_OUTPUT_DIR / f"{patient_slug}_response.txt"

    run_command(
        [
            sys.executable,
            "new_patient_pipeline.py",
            "--patient-input",
            str(patient_csv_path),
            "--output-dir",
            str(PROMPT_OUTPUT_DIR),
        ]
    )

    run_command(
        [
            sys.executable,
            "ollama_explainer.py",
            "--prompt-path",
            str(prompt_path),
            "--output-path",
            str(llm_output_path),
            "--model",
            DEFAULT_LLM_MODEL,
        ]
    )

    if not llm_output_path.exists():
        raise HTTPException(
            status_code=500,
            detail="El pipeline terminó, pero no se generó el archivo con la respuesta del LLM.",
        )

    llm_response = llm_output_path.read_text(encoding="utf-8").strip()

    return {
        "status": "success",
        "paciente": merged_row["nombre_completo"],
        "combined_csv_path": str(patient_csv_path),
        "prompt_path": str(prompt_path),
        "llm_output": llm_response,
    }


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
