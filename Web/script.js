const form = document.getElementById("patient-form");
const fileInput = document.getElementById("csv-file");
const fileDropArea = document.getElementById("file-drop-area");
const fileMessage = fileDropArea.querySelector(".file-msg");
const loadingState = document.getElementById("loading-state");
const resultSection = document.getElementById("result-section");
const llmOutput = document.getElementById("llm-output");
const resetButton = document.getElementById("btn-reset");
const submitButton = document.getElementById("btn-submit");
const tabNewPatient = document.getElementById("tab-new-patient");
const tabHistory = document.getElementById("tab-history");
const newPatientPanel = document.getElementById("new-patient-panel");
const historySection = document.getElementById("history-section");
const historyList = document.getElementById("history-list");
const historyDetail = document.getElementById("history-detail");
const refreshHistoryButton = document.getElementById("btn-refresh-history");

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

let selectedHistoryId = null;

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

function formatDate(value) {
  if (!value) {
    return "Fecha no disponible";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("es-ES", {
    dateStyle: "short",
    timeStyle: "short",
  });
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

function renderCommentEditor(requestId, comment) {
  return `
    <div class="feedback-panel" data-feedback-for="${escapeHtml(requestId)}">
      <label for="feedback-${escapeHtml(requestId)}">Comentario clínico / feedback</label>
      <textarea id="feedback-${escapeHtml(requestId)}" rows="4" placeholder="Añade o actualiza la decisión clínica tomada...">${escapeHtml(comment || "")}</textarea>
      <div class="feedback-actions">
        <span class="feedback-status" aria-live="polite"></span>
        <button type="button" class="btn-secondary btn-save-feedback" data-request-id="${escapeHtml(requestId)}">Guardar comentario</button>
      </div>
    </div>
  `;
}

function renderClinicalDashboard(data, options = {}) {
  const probability = Number(data.predicted_probability || 0);
  const probabilityPercent = `${(probability * 100).toFixed(2)}%`;
  const riskKey = normalizeRisk(data.predicted_risk_group);
  const config = riskConfig[riskKey];
  const progress = Math.min(100, Math.max(0, probability * 100));
  const plan = treatmentPlan(riskKey);
  const commentEditor = options.includeCommentEditor
    ? renderCommentEditor(data.request_id, data.comentario_medico)
    : "";
  const createdAt = data.created_at
    ? `<span class="history-date">${escapeHtml(formatDate(data.created_at))}</span>`
    : "";

  return `
<div class="clinical-dashboard ${config.className}">
  <div class="patient-strip">
    <div>
      <span class="eyebrow">Paciente ${createdAt}</span>
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

  ${commentEditor}
</div>
  `;
}

function renderResult(data) {
  llmOutput.innerHTML = renderClinicalDashboard(data, { includeCommentEditor: true });
  resultSection.classList.remove("hidden");
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  bindFeedbackButtons(llmOutput);
}

function renderError(message) {
  llmOutput.innerHTML = `<strong>Error:</strong><br><br>${escapeHtml(message)}`;
  resultSection.classList.remove("hidden");
}

function showNewPatientView() {
  tabNewPatient.classList.add("is-active");
  tabHistory.classList.remove("is-active");
  newPatientPanel.classList.remove("hidden");
  resultSection.classList.toggle("hidden", !llmOutput.innerHTML.trim());
  historySection.classList.add("hidden");
}

async function showHistoryView() {
  tabHistory.classList.add("is-active");
  tabNewPatient.classList.remove("is-active");
  newPatientPanel.classList.add("hidden");
  resultSection.classList.add("hidden");
  historySection.classList.remove("hidden");
  await loadHistory();
}

function updateSelectedFile() {
  const file = fileInput.files[0];
  fileMessage.textContent = file ? `Archivo seleccionado: ${file.name}` : defaultFileMessage;
}

async function loadHistory() {
  historyList.innerHTML = `<p class="empty-state">Cargando historial...</p>`;
  try {
    const response = await fetch("/api/historial");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "No se pudo cargar el historial.");
    }
    renderHistoryList(payload.items || []);
  } catch (error) {
    historyList.innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
  }
}

function renderHistoryList(items) {
  if (!items.length) {
    historyList.innerHTML = `<p class="empty-state">Todavía no hay pacientes guardados.</p>`;
    historyDetail.innerHTML = `<p class="empty-state">Cuando analices un paciente, aparecerá aquí para futuras consultas.</p>`;
    return;
  }

  historyList.innerHTML = items
    .map((item) => {
      const riskKey = normalizeRisk(item.predicted_risk_group);
      const isActive = item.request_id === selectedHistoryId ? "is-selected" : "";
      return `
        <button type="button" class="history-item ${isActive}" data-request-id="${escapeHtml(item.request_id)}">
          <span>
            <strong>${escapeHtml(item.paciente)}</strong>
            <small>${escapeHtml(formatDate(item.created_at))}</small>
          </span>
          <em class="${riskConfig[riskKey].className}">${riskConfig[riskKey].label}</em>
        </button>
      `;
    })
    .join("");

  historyList.querySelectorAll(".history-item").forEach((button) => {
    button.addEventListener("click", () => loadHistoryDetail(button.dataset.requestId));
  });
}

async function loadHistoryDetail(requestId) {
  selectedHistoryId = requestId;
  historyDetail.innerHTML = `<p class="empty-state">Cargando informe...</p>`;
  try {
    const response = await fetch(`/api/historial/${encodeURIComponent(requestId)}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "No se pudo cargar el paciente.");
    }
    renderHistoryDetail(payload.item);
    historyList.querySelectorAll(".history-item").forEach((button) => {
      button.classList.toggle("is-selected", button.dataset.requestId === requestId);
    });
  } catch (error) {
    historyDetail.innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
  }
}

function renderHistoryDetail(item) {
  const formData = item.formulario || {};
  historyDetail.innerHTML = `
    <div class="history-meta">
      <span>ID: ${escapeHtml(item.request_id)}</span>
      <span>Edad: ${escapeHtml(formData.edad ?? "N/D")}</span>
      <span>Género: ${escapeHtml(formData.genero ?? "N/D")}</span>
      <span>Actualizado: ${escapeHtml(formatDate(item.updated_at || item.created_at))}</span>
    </div>
    ${renderClinicalDashboard(item, { includeCommentEditor: true })}
  `;
  bindFeedbackButtons(historyDetail);
}

function bindFeedbackButtons(root) {
  root.querySelectorAll(".btn-save-feedback").forEach((button) => {
    button.addEventListener("click", async () => {
      const requestId = button.dataset.requestId;
      const panel = button.closest(".feedback-panel");
      const textarea = panel.querySelector("textarea");
      const status = panel.querySelector(".feedback-status");
      button.disabled = true;
      status.textContent = "Guardando...";

      try {
        const response = await fetch(`/api/historial/${encodeURIComponent(requestId)}/comentario`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ comentario: textarea.value }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "No se pudo guardar el comentario.");
        }
        status.textContent = "Comentario guardado.";
        if (payload.item) {
          selectedHistoryId = payload.item.request_id;
          await loadHistory();
        }
      } catch (error) {
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });
  });
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
  llmOutput.innerHTML = "";
  showNewPatientView();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

tabNewPatient.addEventListener("click", showNewPatientView);
tabHistory.addEventListener("click", showHistoryView);
refreshHistoryButton.addEventListener("click", loadHistory);
