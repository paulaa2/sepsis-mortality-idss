from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from pypdf import PdfReader


CORE_FIELDS = [
    "heart_rate_max",
    "mbp_min",
    "temperature_max",
    "resp_rate_max",
    "hematocrit_max",
    "wbc_max",
    "creatinine_max",
    "bun_max",
    "sodium_min",
    "glucose_max",
    "urineoutput",
    "gcs_eyes",
    "gcs_verbal",
    "gcs_motor",
]

NUM = r"([-+]?\d+(?:[,.]\d+)?)"


@dataclass(frozen=True)
class ExtractionRule:
    fields: tuple[str, ...]
    patterns: tuple[str, ...]
    transform: str = "number"


RULES: tuple[ExtractionRule, ...] = (
    ExtractionRule(
        fields=("heart_rate_min", "heart_rate_max"),
        patterns=(
            rf"(?:frecuencia\s+cardiaca|fc|heart\s+rate|hr)\s*[:=]?\s*{NUM}\s*(?:lpm|bpm)?",
            rf"(?:fc|hr)\s+{NUM}\s*(?:lpm|bpm)?",
        ),
    ),
    ExtractionRule(
        fields=("temperature_min", "temperature_max"),
        patterns=(
            rf"(?:temperatura|temp\.?|temperature)\s*[:=]?\s*{NUM}\s*(?:c|celsius|degrees|deg)?",
        ),
    ),
    ExtractionRule(
        fields=("resp_rate_min", "resp_rate_max"),
        patterns=(
            rf"(?:frecuencia\s+respiratoria|fr|respiratory\s+rate|rr)\s*[:=]?\s*{NUM}\s*(?:rpm|irpm)?",
        ),
    ),
    ExtractionRule(
        fields=("hematocrit_min", "hematocrit_max"),
        patterns=(
            rf"(?:hematocrito|hto|hct|hematocrit)\s*[:=]?\s*{NUM}\s*%?",
        ),
    ),
    ExtractionRule(
        fields=("wbc_min", "wbc_max"),
        patterns=(
            rf"(?:leucocitos|leucocytes|wbc|white\s+blood\s+cells)\s*[:=]?\s*{NUM}",
        ),
    ),
    ExtractionRule(
        fields=("creatinine_min", "creatinine_max"),
        patterns=(
            rf"(?:creatinina|creatinine)\s*[:=]?\s*{NUM}\s*(?:mg/dl|umol/l)?",
        ),
    ),
    ExtractionRule(
        fields=("bun_min", "bun_max"),
        patterns=(
            rf"(?:bun|urea|nitrogeno\s+ureico)\s*[:=]?\s*{NUM}\s*(?:mg/dl)?",
        ),
    ),
    ExtractionRule(
        fields=("sodium_min", "sodium_max"),
        patterns=(
            rf"(?:\bsodio\b|\bna\+?\b|\bsodium\b)\s*[:=]?\s*{NUM}\s*(?:mmol/l|meq/l)?",
        ),
    ),
    ExtractionRule(
        fields=("glucose_min", "glucose_max"),
        patterns=(
            rf"(?:glucosa|glycemia|glucose)\s*[:=]?\s*{NUM}\s*(?:mg/dl|mmol/l)?",
        ),
    ),
    ExtractionRule(
        fields=("urineoutput",),
        patterns=(
            rf"(?:diuresis|diuresi|urine\s+output|uo)\s*[:=]?\s*{NUM}\s*(?:ml|ml/24h)?",
        ),
    ),
    ExtractionRule(
        fields=("gcs_eyes",),
        patterns=(rf"(?:gcs\s+ojos|glasgow\s+ojos|ocular|eyes?)\s*[:=]?\s*{NUM}",),
    ),
    ExtractionRule(
        fields=("gcs_verbal",),
        patterns=(rf"(?:gcs\s+verbal|glasgow\s+verbal|verbal)\s*[:=]?\s*{NUM}",),
    ),
    ExtractionRule(
        fields=("gcs_motor",),
        patterns=(rf"(?:gcs\s+motor|glasgow\s+motor|motor)\s*[:=]?\s*{NUM}",),
    ),
    ExtractionRule(
        fields=("apsiii",),
        patterns=(rf"(?:apsiii|aps\s*iii)\s*[:=]?\s*{NUM}",),
    ),
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def read_template_row(template_csv_path: Path) -> tuple[list[str], dict[str, str]]:
    with template_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        row = next(reader)
        return list(reader.fieldnames or []), dict(row)


def normalise_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def parse_number(raw_value: str) -> float | None:
    cleaned = raw_value.strip().replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def format_csv_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}".replace(".", ",")


def first_number_match(text: str, patterns: Iterable[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return parse_number(match.group(1))
    return None


def extract_values_from_text(text: str) -> dict[str, float | int | str]:
    clean_text = normalise_text(text)
    values: dict[str, float | int | str] = {}

    for rule in RULES:
        value = first_number_match(clean_text, rule.patterns)
        if value is None:
            continue
        for field in rule.fields:
            values[field] = value

    blood_pressure = re.search(
        rf"(?:ta|pa|presion\s+arterial|blood\s+pressure)\s*[:=]?\s*{NUM}\s*/\s*{NUM}",
        clean_text,
        flags=re.IGNORECASE,
    )
    if blood_pressure:
        systolic = parse_number(blood_pressure.group(1))
        diastolic = parse_number(blood_pressure.group(2))
        if systolic is not None and diastolic is not None:
            mean_pressure = (systolic + 2 * diastolic) / 3
            values["mbp_min"] = mean_pressure
            values["mbp_max"] = mean_pressure

    mean_pressure = first_number_match(
        clean_text,
        (
            rf"(?:pam|presion\s+arterial\s+media|mean\s+arterial\s+pressure|map)\s*[:=]?\s*{NUM}",
        ),
    )
    if mean_pressure is not None:
        values["mbp_min"] = mean_pressure
        values["mbp_max"] = mean_pressure

    age = first_number_match(clean_text, (rf"(?:edad|age)\s*[:=]?\s*{NUM}",))
    if age is not None and 0 <= age <= 120:
        values["admission_age"] = age

    if re.search(r"\b(sepsis|sepsia|septicemia)\b", clean_text, flags=re.IGNORECASE):
        values["sepsis3"] = 1

    gender_match = re.search(
        r"(?:sexo|genero|gender)\s*[:=]?\s*(masculino|femenino|hombre|mujer|male|female|m|f)\b",
        clean_text,
        flags=re.IGNORECASE,
    )
    if gender_match:
        values["gender"] = gender_to_code(gender_match.group(1))

    return values


def gender_to_code(value: str) -> int:
    lower = value.strip().lower()
    if lower in {"masculino", "hombre", "male", "m"}:
        return 1
    if lower in {"femenino", "mujer", "female", "f"}:
        return 0
    return 2


def apply_form_context(row: dict[str, str], form_context: dict[str, object] | None) -> None:
    if not form_context:
        return

    if form_context.get("edad") not in (None, ""):
        row["admission_age"] = format_csv_value(float(form_context["edad"]))
    if form_context.get("genero") not in (None, ""):
        row["gender"] = str(gender_to_code(str(form_context["genero"])))
    if form_context.get("etnia") not in (None, ""):
        # The model was trained with encoded categorical values. For free text,
        # keep the template code unless the form already sends a numeric code.
        raw_ethnicity = str(form_context["etnia"]).strip()
        if raw_ethnicity.isdigit():
            row["ethnicity"] = raw_ethnicity


def write_patient_csv(output_csv_path: Path, columns: list[str], row: dict[str, str]) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def build_patient_csv_from_pdf(
    pdf_path: Path,
    output_csv_path: Path,
    template_csv_path: Path,
    form_context: dict[str, object] | None = None,
    extracted_text_path: Path | None = None,
) -> dict[str, object]:
    columns, row = read_template_row(template_csv_path)
    text = extract_text_from_pdf(pdf_path)

    if extracted_text_path is not None:
        extracted_text_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_text_path.write_text(text, encoding="utf-8")

    if not text:
        raise ValueError(
            "No se ha podido extraer texto del PDF. Si es un documento escaneado, hace falta anadir OCR."
        )

    values = extract_values_from_text(text)
    for field, value in values.items():
        if field in row:
            row[field] = format_csv_value(value)

    row["subject_id"] = row.get("subject_id") or str(int(uuid4().int % 10_000_000))
    row["hadm_id"] = row.get("hadm_id") or str(int(uuid4().int % 10_000_000))
    row["stay_id"] = row.get("stay_id") or str(int(uuid4().int % 10_000_000))

    apply_form_context(row, form_context)
    write_patient_csv(output_csv_path, columns, row)

    extracted_fields = sorted(field for field in values if field in columns)
    missing_core_fields = sorted(field for field in CORE_FIELDS if field not in extracted_fields)
    return {
        "source_type": "pdf",
        "pdf_path": str(pdf_path),
        "generated_csv_path": str(output_csv_path),
        "extracted_text_path": str(extracted_text_path) if extracted_text_path else None,
        "extracted_fields": extracted_fields,
        "extracted_field_count": len(extracted_fields),
        "missing_core_fields": missing_core_fields,
        "text_characters": len(text),
    }
