const form = document.getElementById("patient-form");
const fileInput = document.getElementById("csv-file");
const fileDropArea = document.getElementById("file-drop-area");
const fileMessage = fileDropArea.querySelector(".file-msg");
const loadingState = document.getElementById("loading-state");
const resultSection = document.getElementById("result-section");
const llmOutput = document.getElementById("llm-output");
const resetButton = document.getElementById("btn-reset");
const submitButton = document.getElementById("btn-submit");

const defaultFileMessage = "Arrastra tu archivo CSV aquí o haz clic para buscar";

const riskConfig = {
  low: {
    label: "Riesgo bajo",
    className: "risk-low",
    summary: "Seguimiento clínico y reevaluación si cambia la situación.",
  },
  medium: {
    label: "Riesgo medio",
    className: "risk-medium",
    summary: "Vigilancia estrecha y completar evaluación infecciosa.",
  },
  high: {
    label: "Riesgo alto",
    className: "risk-high",
    summary: "Reevaluación urgente y posible escalada asistencial.",
  },
};

function setLoading(isLoading) {
  loadingState.classList.toggle("hidden", !isLoading);
  submitButton.disabled = isLoading;
  submitButton.style.opacity = isLoading ? "0.7" : "1";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function normalizeRisk(riskGroup) {
  const key = String(riskGroup || "").toLowerCase();
  return riskConfig[key] ? key : "medium";
}

function treatmentPlan(riskKey) {
  if (riskKey === "high") {
    return [
      ["Hacer ahora", "Cultivos si no retrasan, antibiótico precoz, lactato y reevaluación hemodinámica."],
      ["Valorar", "Fluidoterapia si hipotensión/hipoperfusión; vasopresores si PAM < 65 pese a fluidos."],
      ["Escalada", "Avisar UCI/equipo responsable si deterioro, shock o compromiso respiratorio."],
    ];
  }

  if (riskKey === "medium") {
    return [
      ["Hacer ahora", "Confirmar foco, tomar cultivos y medir lactato si no está disponible."],
      ["Valorar", "Antibiótico precoz si sospecha clínica de sepsis; fluidos solo si hipoperfusión."],
      ["Escalada", "Reevaluar pronto si empeora PAM, diuresis, estado mental o respiración."],
    ];
  }

  return [
    ["Hacer ahora", "Monitorización, revisión clínica y completar datos clave si faltan."],
    ["Valorar", "Antibiótico solo si hay sospecha clínica de infección/sepsis."],
    ["No rutinario", "Fluidoterapia agresiva, vasopresores o escalada si no hay hipoperfusión/deterioro."],
  ];
}

function formatClinicalText(text) {
  const cleanText = String(text || "Sin explicación clínica disponible.")
    .replace(/\*\*/g, "")
    .replace(/Aquí tienes.*?\n/gi, "")
    .replace(/Frases preferidas:[\s\S]*$/gi, "")
    .trim();

  return cleanText
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 8)
    .map((line) => `<p>${escapeHtml(line).replace(/^[-*]\s*/, "")}</p>`)
    .join("");
}

function renderResult(data) {
  const probability = Number(data.predicted_probability || 0);
  const probabilityPercent = `${(probability * 100).toFixed(2)}%`;
  const riskKey = normalizeRisk(data.predicted_risk_group);
  const config = riskConfig[riskKey];
  const progress = Math.min(100, Math.max(0, probability * 100));
  const plan = treatmentPlan(riskKey);

  llmOutput.innerHTML = `
<div class="clinical-dashboard ${config.className}">
  <div class="patient-strip">
    <div>
      <span class="eyebrow">Paciente</span>
      <h3>${escapeHtml(data.paciente)}</h3>
    </div>
    <span class="risk-pill">${config.label}</span>
  </div>

  <div class="metric-grid">
    <article class="metric-card">
      <span>Probabilidad de mortalidad</span>
      <strong>${probabilityPercent}</strong>
      <div class="risk-bar">
        <div style="width: ${progress}%"></div>
      </div>
      <small>${config.summary}</small>
    </article>
    <article class="metric-card">
      <span>Fenotipo clínico</span>
      <strong>${escapeHtml(data.cluster_label)}</strong>
      <small>Cluster asignado por similitud clínica.</small>
    </article>
  </div>

  <div class="treatment-panel">
    <h4>Conducta recomendada</h4>
    <div class="action-list">
      ${plan
        .map(
          ([title, body]) => `
          <div class="action-chip">
            <strong>${escapeHtml(title)}</strong>
            <span>${escapeHtml(body)}</span>
          </div>
        `,
        )
        .join("")}
    </div>
  </div>

  <div class="explanation-panel">
    <h4>Lectura clínica del LLM</h4>
    ${formatClinicalText(data.explanation)}
  </div>
</div>
  `;

  resultSection.classList.remove("hidden");
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderError(message) {
  llmOutput.innerHTML = `<strong>Error:</strong><br><br>${escapeHtml(message)}`;
  resultSection.classList.remove("hidden");
}

function updateSelectedFile() {
  const file = fileInput.files[0];
  fileMessage.textContent = file ? `Archivo seleccionado: ${file.name}` : defaultFileMessage;
}

fileInput.addEventListener("change", updateSelectedFile);

["dragenter", "dragover"].forEach((eventName) => {
  fileDropArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    fileDropArea.classList.add("is-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  fileDropArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    fileDropArea.classList.remove("is-active");
  });
});

fileDropArea.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (files.length > 0) {
    fileInput.files = files;
    updateSelectedFile();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) {
    renderError("Debes adjuntar un archivo CSV del paciente.");
    return;
  }

  const formData = new FormData(form);
  formData.append("archivo", fileInput.files[0]);

  resultSection.classList.add("hidden");
  setLoading(true);

  try {
    const response = await fetch("/api/analizar", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "No se pudo completar el análisis.");
    }

    renderResult(payload);
  } catch (error) {
    renderError(error.message);
  } finally {
    setLoading(false);
  }
});

resetButton.addEventListener("click", () => {
  form.reset();
  fileMessage.textContent = defaultFileMessage;
  resultSection.classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
});
