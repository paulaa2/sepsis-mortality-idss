import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

# Configurar CORS para permitir que el frontend se comunique con el backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"

def simular_pipeline_modelo(csv_contents: bytes):
    """
    Aquí deberías instanciar tu modelo de ML real.
    """
    # TODO: Cargar tu modelo y hacer predicción con csv_contents
    return "Paciente con nivel de riesgo moderado de padecer la enfermedad (70% prob)."

def simular_clustering(csv_contents: bytes):
    """
    Aquí deberías instanciar tu modelo de clustering real.
    """
    # TODO: Aplicar K-means u otro
    return "Cluster 3 (Historial clínico inflamatorio y comorbilidades asociadas)."

@app.post("/api/analizar")
async def analizar_paciente(
    nombre: str = Form(...),
    apellido: str = Form(...),
    edad: int = Form(...),
    altura: float = Form(...),
    peso: float = Form(...),
    genero: str = Form(...),
    etnia: str = Form(...),
    archivo: UploadFile = File(...)
):
    # 1. Leer el archivo CSV
    csv_bytes = await archivo.read()
    
    # 2. Ejecutar Modelo y Clustering (SIMULADOS en este ejemplo)
    resultado_modelo = simular_pipeline_modelo(csv_bytes)
    resultado_clustering = simular_clustering(csv_bytes)
    
    # 3. Leer Base de Conocimientos que ya está en el sistema
    kb_path = "base_conocimientos.txt" # Ruta relativa o absoluta
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            base_conocimientos = f.read()
    else:
        base_conocimientos = "Base de conocimiento: Los pacientes del cluster 3 con riesgo moderado deben ser monitorizados cada 6 meses, y realizar exámenes bioquímicos periódicos."
    
    # 4. Construir el Prompt para el LLM
    prompt = f"""
    Actúa como un médico experto analizando el siguiente caso clínico.
    
    Datos del Paciente:
    - Nombre: {nombre} {apellido}
    - Edad: {edad} años
    - Altura: {altura} cm
    - Peso: {peso} kg
    - Género: {genero}
    - Etnia: {etnia}
    
    Resultados del Pipeline de Inteligencia Artificial:
    - Predicción del Modelo: {resultado_modelo}
    - Clustering: {resultado_clustering}
    
    Base de conocimientos médica del hospital:
    {base_conocimientos}
    
    Por favor, genera un informe detallado, un diagnóstico presuntivo y recomendaciones profesionales para este paciente basándote estrictamente en los datos proporcionados.
    """
    
    # 5. Enviar el Prompt a Ollama (Local)
    payload = {
        "model": "llama3", # TODO: Cambia esto si usas otro modelo en Ollama, ej: mistral, qwen
        "prompt": prompt,
        "stream": False 
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=5)
        if response.status_code == 200:
            llm_response = response.json().get("response", "No se obtuvo respuesta del LLM.")
        else:
            llm_response = f"Error al procesar con Ollama. Código de estado HTTP: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        # Como Ollama no está instalado en este ordenador, devolvemos una respuesta de simulación muy currada:
        llm_response = f"""⚠️ *Nota: No se pudo conectar a la base de Ollama (Puerto 11434).*
***[SIMULACIÓN DE INFORME MÉDICO GENERADO POR LLM]***

**INFORME MÉDICO DE INTELIGENCIA ARTIFICIAL**

**Paciente:** {nombre} {apellido} ({edad} años, {genero}, {etnia})
**Antropometría:** {peso} kg / {altura} cm

**Análisis de Resultados Algorítmicos:**
En base a los datos biométricos recolectados y la extracción de características del archivo CSV, se observa que el paciente se clasifica en el **{resultado_clustering}**. Esto indica una posible predisposición a factores sistémicos que concuerdan con la predicción del modelo: _{resultado_modelo}_.

**Diagnóstico Presuntivo:**
Pacientes con la configuración fenotípica y las métricas computacionales arrojadas presentan un cuadro que requiere atención clínica preventiva.

**Plan de Acción Inteligente (basado en protocolos):**
1. Iniciar monitorización estrecha cada 6 meses, según directrices del cluster número 3.
2. Evaluar pruebas cardiovasculares y bioquímicas rutinarias.
3. Ajuste nutricional dadas las variables de peso y altura.

_Firma del modelo: Sistema de Diagnóstico IA_"""

    # Retornar datos al frontend
    return {
        "status": "success",
        "paciente": f"{nombre} {apellido}",
        "llm_output": llm_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
