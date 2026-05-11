const form = document.getElementById("patient-form");
const resultSection = document.getElementById("result-section");
const loadingState = document.getElementById("loading-state");
const llmOutput = document.getElementById("llm-output");
const fileInput = document.getElementById("csv-file");
const fileDropArea = document.getElementById("file-drop-area");
const fileMessage = fileDropArea.querySelector(".file-msg");
const submitButton = document.getElementById("btn-submit");
const resetButton = document.getElementById("btn-reset");

function setLoadingState(isLoading) {
    loadingState.classList.toggle("hidden", !isLoading);
    submitButton.disabled = isLoading;
    submitButton.querySelector("span").textContent = isLoading ? "Analizando..." : "Analizar Paciente";
}

function renderResult(text) {
    llmOutput.textContent = text;
    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderError(message) {
    renderResult(`Error durante el analisis:\n\n${message}`);
}

function updateFileMessage() {
    const file = fileInput.files?.[0];
    fileMessage.textContent = file
        ? `Archivo listo: ${file.name}`
        : "Arrastra tu archivo CSV aqui o haz clic para buscar";
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput.files?.[0];
    if (!file) {
        renderError("Selecciona un CSV antes de lanzar el analisis.");
        return;
    }

    setLoadingState(true);
    resultSection.classList.add("hidden");

    try {
        const formData = new FormData(form);
        formData.append("archivo", file);

        const response = await fetch("/api/analizar", {
            method: "POST",
            body: formData,
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "No se pudo completar el pipeline.");
        }

        renderResult(payload.llm_output || "El LLM no devolvio contenido.");
    } catch (error) {
        renderError(error.message || "Ha ocurrido un error inesperado.");
    } finally {
        setLoadingState(false);
    }
});

fileInput.addEventListener("change", updateFileMessage);

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
    const files = event.dataTransfer?.files;
    if (!files?.length) {
        return;
    }

    const dataTransfer = new DataTransfer();
    Array.from(files).forEach((file) => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;
    updateFileMessage();
});

resetButton.addEventListener("click", () => {
    form.reset();
    updateFileMessage();
    resultSection.classList.add("hidden");
    llmOutput.textContent = "";
    window.scrollTo({ top: 0, behavior: "smooth" });
});
