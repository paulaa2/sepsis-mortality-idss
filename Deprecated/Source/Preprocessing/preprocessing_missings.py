from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS_MISSINGS = [
    "analisis_missings.py",
    "quitar_variables_missings.py",
    "limpiar_negativos.py",
    "quitar_variables_fecha.py",
    "analisis_e_imputacion_knn.py",
]


def run_script(script_path: Path) -> None:
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Fallo al ejecutar {script_path.name} (codigo {result.returncode})."
        )


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    for script_name in SCRIPTS_MISSINGS:
        script_path = base_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"No se encontro el script requerido: {script_path.name}")

    print("Iniciando bloque de tratamiento de missings...")
    for idx, script_name in enumerate(SCRIPTS_MISSINGS, start=1):
        script_path = base_dir / script_name
        print(f"[{idx}/{len(SCRIPTS_MISSINGS)}] Ejecutando {script_name}...")
        run_script(script_path)

    print("Bloque de tratamiento de missings completado.")


if __name__ == "__main__":
    main()
