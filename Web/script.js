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

function setLoading(isLoading) {
  loadingState.classList.toggle("hidden", !isLoading);
  submitButton.disabled = isLoading;
  submitButton.style.opacity = isLoading ? "0.7" : "1";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderResult(data) {
  const probability = Number(data.predicted_probability || 0);
  const probabilityPercent = `${(probability * 100).toFixed(2)}%`;
  const safeExplanation = escapeHtml(data.explanation || "Sin explicación disponible.");

  llmOutput.innerHTML = `
<strong>Paciente:</strong> ${escapeHtml(data.paciente)}<br>
<strong>Probabilidad de mortalidad:</strong> ${probabilityPercent}<br>
<strong>Grupo de riesgo:</strong> ${escapeHtml(data.predicted_risk_group)}<br>
<strong>Cluster asignado:</strong> ${escapeHtml(data.cluster_label)}<br><br>
<strong>Explicación clínica:</strong><br><br>
${safeExplanation.replaceAll("\n", "<br>")}
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
