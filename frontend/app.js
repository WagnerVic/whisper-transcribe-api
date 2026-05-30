const API_URL = "http://localhost:8000";

const $ = (sel) => document.querySelector(sel);

const els = {
    statusBadge:    $("#statusBadge"),
    statusText:     $(".status-text"),
    uploadSection:  $("#uploadSection"),
    uploadZone:     $("#uploadZone"),
    selectFileBtn:  $("#selectFileBtn"),
    fileInput:      $("#fileInput"),
    fileInfo:       $("#fileInfo"),
    fileName:       $("#fileName"),
    fileSize:       $("#fileSize"),
    removeFileBtn:  $("#removeFileBtn"),
    transcribeBtn:  $("#transcribeBtn"),
    cancelBtn:      $("#cancelBtn"),
    processingSection: $("#processingSection"),
    processingTitle: $("#processingTitle"),
    processingHint: $("#processingHint"),
    progressBar:    $("#progressBar"),
    progressLabel:  $("#progressLabel"),
    logEntries:     $("#logEntries"),
    resultSection:  $("#resultSection"),
    resultText:     $("#resultText"),
    metaLang:       $("#metaLang"),
    metaDuration:   $("#metaDuration"),
    metaTime:       $("#metaTime"),
    copyBtn:        $("#copyBtn"),
    newBtn:         $("#newBtn"),
    languageSelect: $("#languageSelect"),
    speedInput:     $("#speedInput"),
    modelSelect:    $("#modelSelect"),
    diarizeSelect:  $("#diarizeSelect"),
    projectName:    $("#projectName"),
};

let selectedFile = null;
let startTime = null;
let abortController = null;

// --- Health Check ---
async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        els.statusBadge.className = "status-badge online";
        els.statusText.textContent = `GPU: ${data.device.toUpperCase()}`;
    } catch {
        els.statusBadge.className = "status-badge offline";
        els.statusText.textContent = "API offline";
    }
}

// --- File Selection ---
els.selectFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    els.fileInput.click();
});

els.uploadZone.addEventListener("click", () => {
    els.fileInput.click();
});

els.fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// --- Drag & Drop ---
els.uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.uploadZone.classList.add("drag-over");
});

els.uploadZone.addEventListener("dragleave", () => {
    els.uploadZone.classList.remove("drag-over");
});

els.uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    els.fileName.textContent = file.name;
    els.fileSize.textContent = formatSize(file.size);

    els.uploadSection.classList.add("hidden");
    els.fileInfo.classList.remove("hidden");
}

els.removeFileBtn.addEventListener("click", resetToUpload);

function resetToUpload() {
    selectedFile = null;
    els.fileInput.value = "";
    els.fileInfo.classList.add("hidden");
    els.resultSection.classList.add("hidden");
    els.processingSection.classList.add("hidden");
    els.uploadSection.classList.remove("hidden");
}

// --- Add log entry ---
function addLog(message, type = "info") {
    const elapsed = startTime ? ((Date.now() - startTime) / 1000).toFixed(1) : "0.0";
    const entry = document.createElement("div");
    entry.className = `log-entry ${type === "success" ? "log-success" : ""} ${type === "error" ? "log-error" : ""}`;
    entry.innerHTML = `
        <span class="log-time">${elapsed}s</span>
        <span class="log-msg">${message}</span>
    `;
    els.logEntries.appendChild(entry);
    els.logEntries.scrollTop = els.logEntries.scrollHeight;
}

function setProgress(percent) {
    els.progressBar.style.width = `${percent}%`;
    els.progressLabel.textContent = `${percent}%`;
}

// --- Transcribe with SSE ---
els.transcribeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    els.fileInfo.classList.add("hidden");
    els.processingSection.classList.remove("hidden");
    els.logEntries.innerHTML = "";
    setProgress(0);
    startTime = Date.now();

    abortController = new AbortController();

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("stream", "true");
    formData.append("model_size", els.modelSelect.value);
    formData.append("diarize", els.diarizeSelect.value);

    const lang = els.languageSelect.value;
    if (lang) formData.append("language", lang);

    const speed = els.speedInput.value;
    formData.append("speed_up", speed);

    const projName = els.projectName.value.trim();
    if (projName) formData.append("project_name", projName);

    addLog("📤 Enviando arquivo para a API...");

    try {
        const res = await fetch(`${API_URL}/transcribe`, {
            method: "POST",
            body: formData,
            signal: abortController.signal,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Erro na transcrição");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const parts = buffer.split("\n\n");
            buffer = parts.pop();

            for (const part of parts) {
                const lines = part.split("\n");
                let eventType = "message";
                let eventData = "";

                for (const line of lines) {
                    if (line.startsWith("event: ")) eventType = line.slice(7);
                    else if (line.startsWith("data: ")) eventData = line.slice(6);
                }

                if (!eventData) continue;

                try {
                    const data = JSON.parse(eventData);

                    if (eventType === "log") {
                        const isSuccess = data.step?.endsWith("_done") || data.step === "done";
                        addLog(data.message, isSuccess ? "success" : "info");
                        if (data.progress) setProgress(data.progress);

                        if (data.step === "whisper") {
                            els.processingTitle.textContent = "Transcrevendo...";
                            els.processingHint.textContent = "Essa é a etapa mais longa — GPU processando o áudio";
                        }
                    } else if (eventType === "result") {
                        showResult(data);
                    } else if (eventType === "error") {
                        addLog(`❌ Erro: ${data.message}`, "error");
                    }
                } catch {}
            }
        }
    } catch (err) {
        if (err.name === "AbortError") {
            addLog("🛑 Transcrição cancelada pelo usuário", "error");
            setTimeout(() => {
                els.processingSection.classList.add("hidden");
                els.fileInfo.classList.remove("hidden");
            }, 1000);
        } else {
            addLog(`❌ ${err.message}`, "error");
            els.processingSection.classList.add("hidden");
            els.fileInfo.classList.remove("hidden");
            alert(`Erro: ${err.message}`);
        }
    } finally {
        abortController = null;
    }
});

function showResult(data) {
    // Keep processing section visible briefly, then show result
    setTimeout(() => {
        els.processingSection.classList.add("hidden");
        els.resultSection.classList.remove("hidden");

        els.resultText.textContent = data.text;
        els.metaLang.textContent = `Idioma: ${data.language}`;
        els.metaDuration.textContent = `Duração: ${formatDuration(data.duration)}`;
        els.metaTime.textContent = `Processado em ${data.processing_time_seconds}s`;
    }, 1500);
}

// --- Copy ---
els.copyBtn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(els.resultText.textContent);
    const original = els.copyBtn.innerHTML;
    els.copyBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copiado!`;
    setTimeout(() => { els.copyBtn.innerHTML = original; }, 2000);
});

// --- Cancel ---
els.cancelBtn.addEventListener("click", () => {
    if (abortController) {
        abortController.abort();
    }
});

// --- New ---
els.newBtn.addEventListener("click", resetToUpload);

// --- Helpers ---
function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB";
    return (bytes / 1073741824).toFixed(2) + " GB";
}

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// --- Init ---
checkHealth();
setInterval(checkHealth, 15000);

// --- GPU Monitor ---
const gpuEls = {
    barFill: $("#gpuBarFill"),
    value:   $("#gpuValue"),
    util:    $("#gpuUtil"),
    temp:    $("#gpuTemp"),
};

async function updateGpu() {
    try {
        const res = await fetch(`${API_URL}/gpu`);
        const data = await res.json();
        const pct = Math.round((data.vram_used_mb / data.vram_total_mb) * 100);
        const usedGb = (data.vram_used_mb / 1024).toFixed(1);
        const totalGb = (data.vram_total_mb / 1024).toFixed(1);

        gpuEls.barFill.style.width = `${pct}%`;
        gpuEls.barFill.className = `gpu-bar-fill${pct > 80 ? " high" : ""}`;
        gpuEls.value.textContent = `${usedGb}/${totalGb}`;
        gpuEls.util.textContent = `GPU ${data.gpu_util_pct}%`;
        gpuEls.temp.textContent = `${data.temp_c}°C`;
    } catch {}
}

updateGpu();
setInterval(updateGpu, 2000);
