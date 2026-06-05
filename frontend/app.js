const $ = (sel) => document.querySelector(sel);

/* ── Element refs ────────────────────────────────────────────── */
const els = {
    statusBadge:       $("#statusBadge"),
    statusText:        $(".status-text"),
    uploadSection:     $("#uploadSection"),
    uploadZone:        $("#uploadZone"),
    fileCard:          $("#fileCard"),
    selectFileBtn:     $("#selectFileBtn"),
    fileInput:         $("#fileInput"),
    fileName:          $("#fileName"),
    fileSize:          $("#fileSize"),
    removeFileBtn:     $("#removeFileBtn"),
    transcribeBtn:     $("#transcribeBtn"),
    cancelBtn:         $("#cancelBtn"),
    processingSection: $("#processingSection"),
    processingTitle:   $("#processingTitle"),
    processingHint:    $("#processingHint"),
    progressBar:       $("#progressBar"),
    progressLabel:     $("#progressLabel"),
    logEntries:        $("#logEntries"),
    resultSection:     $("#resultSection"),
    resultText:        $("#resultText"),
    metaLang:          $("#metaLang"),
    metaDuration:      $("#metaDuration"),
    metaTime:          $("#metaTime"),
    copyBtn:           $("#copyBtn"),
    newBtn:            $("#newBtn"),
    downloadBar:       $("#downloadBar"),
    languageSelect:    $("#languageSelect"),
    speedInput:        $("#speedInput"),
    modelSelect:       $("#modelSelect"),
    diarizeSelect:     $("#diarizeSelect"),
    projectName:       $("#projectName"),
    historyList:       $("#historyList"),
    refreshHistoryBtn: $("#refreshHistoryBtn"),
    saveSettingsBtn:   $("#saveSettingsBtn"),
    savedMsg:          $("#savedMsg"),
};

let selectedFile = null;
let startTime    = null;
let abortController = null;

/* ── Tab navigation ──────────────────────────────────────────── */
const tabBtns     = document.querySelectorAll(".tab-btn");
const panels      = document.querySelectorAll(".panel");
const tabIndicator = $("#tabIndicator");

function switchTab(name) {
    tabBtns.forEach(btn => btn.classList.toggle("active", btn.dataset.tab === name));
    panels.forEach(p => p.classList.toggle("active", p.id === `panel-${name}`));
    const activeBtn = document.querySelector(`.tab-btn[data-tab="${name}"]`);
    if (activeBtn) moveIndicator(activeBtn);
    if (name === "history") loadHistory();
    if (name === "settings") loadSettingsForm();
}

function moveIndicator(btn) {
    const navRect = btn.parentElement.getBoundingClientRect();
    const btnRect = btn.getBoundingClientRect();
    tabIndicator.style.width = btnRect.width + "px";
    tabIndicator.style.left  = (btnRect.left - navRect.left) + "px";
}

tabBtns.forEach(btn => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

// Init indicator after fonts load
requestAnimationFrame(() => {
    const active = document.querySelector(".tab-btn.active");
    if (active) moveIndicator(active);
});

/* ── Health check ────────────────────────────────────────────── */
async function checkHealth() {
    try {
        const data = await fetch("/health").then(r => r.json());
        els.statusBadge.className = "status-badge online";
        els.statusText.textContent = `GPU: ${data.device.toUpperCase()}`;
    } catch {
        els.statusBadge.className = "status-badge offline";
        els.statusText.textContent = "API offline";
    }
}

/* ── File selection ──────────────────────────────────────────── */
els.selectFileBtn.addEventListener("click", e => { e.stopPropagation(); els.fileInput.click(); });
els.uploadZone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

els.uploadZone.addEventListener("dragover",  e => { e.preventDefault(); els.uploadZone.classList.add("drag-over"); });
els.uploadZone.addEventListener("dragleave", ()  => els.uploadZone.classList.remove("drag-over"));
els.uploadZone.addEventListener("drop", e => {
    e.preventDefault();
    els.uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    els.fileName.textContent = file.name;
    els.fileSize.textContent = fmtSize(file.size);
    els.uploadZone.classList.add("hidden");
    els.fileCard.classList.remove("hidden");
    els.transcribeBtn.disabled = false;
}

els.removeFileBtn.addEventListener("click", resetToUpload);

function resetToUpload() {
    selectedFile = null;
    els.fileInput.value = "";
    els.uploadZone.classList.remove("hidden");
    els.fileCard.classList.add("hidden");
    els.transcribeBtn.disabled = true;
    els.uploadSection.classList.remove("hidden");
    els.resultSection.classList.add("hidden");
    els.processingSection.classList.add("hidden");
    els.downloadBar.classList.add("hidden");
    els.downloadBar.innerHTML = "";
}

/* ── Logs ────────────────────────────────────────────────────── */
function addLog(msg, type = "info") {
    const elapsed = startTime ? ((Date.now() - startTime) / 1000).toFixed(1) : "0.0";
    const row = document.createElement("div");
    row.className = `log-entry${type === "success" ? " log-success" : ""}${type === "error" ? " log-error" : ""}`;
    const t = document.createElement("span"); t.className = "log-time"; t.textContent = elapsed + "s";
    const m = document.createElement("span"); m.className = "log-msg";  m.textContent = msg;
    row.append(t, m);
    els.logEntries.appendChild(row);
    els.logEntries.scrollTop = els.logEntries.scrollHeight;
}

function setProgress(pct) {
    els.progressBar.style.width = pct + "%";
    els.progressLabel.textContent = pct + "%";
}

/* ── Transcribe ──────────────────────────────────────────────── */
els.transcribeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    els.uploadSection.classList.add("hidden");
    els.processingSection.classList.remove("hidden");
    els.processingTitle.textContent = "Processando...";
    els.processingHint.textContent = "Acompanhe o progresso abaixo";
    els.logEntries.innerHTML = "";
    setProgress(0);
    startTime = Date.now();
    startGpuPolling(true);

    abortController = new AbortController();

    const fd = new FormData();
    fd.append("file", selectedFile);
    fd.append("stream", "true");
    fd.append("model_size",  els.modelSelect.value);
    fd.append("diarize",     els.diarizeSelect.value);
    fd.append("speed_up",    els.speedInput.value);
    if (els.languageSelect.value) fd.append("language", els.languageSelect.value);
    const proj = els.projectName.value.trim();
    if (proj) fd.append("project_name", proj);

    addLog("📤 Enviando arquivo...");

    try {
        const res = await fetch("/transcribe", { method:"POST", body:fd, signal:abortController.signal });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Erro na API"); }

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop();

            for (const part of parts) {
                let eventType = "message", eventData = "";
                for (const line of part.split("\n")) {
                    if (line.startsWith("event: ")) eventType = line.slice(7);
                    else if (line.startsWith("data: ")) eventData = line.slice(6);
                }
                if (!eventData) continue;
                try {
                    const data = JSON.parse(eventData);
                    if (eventType === "log") {
                        const ok = data.step?.endsWith("_done") || data.step === "done";
                        addLog(data.message, ok ? "success" : "info");
                        if (data.progress) setProgress(data.progress);
                        if (data.step === "whisper") {
                            els.processingTitle.textContent = "Transcrevendo com Whisper...";
                            els.processingHint.textContent = "GPU processando — a etapa mais longa";
                        }
                    } else if (eventType === "result") {
                        showResult(data);
                    } else if (eventType === "files") {
                        showDownloads(data);
                    } else if (eventType === "error") {
                        addLog(`❌ ${data.message}`, "error");
                    }
                } catch {}
            }
        }
    } catch (err) {
        if (err.name === "AbortError") {
            addLog("🛑 Cancelado pelo usuário", "error");
            setTimeout(resetToUpload, 1200);
        } else {
            addLog(`❌ ${err.message}`, "error");
            setTimeout(resetToUpload, 2000);
        }
    } finally {
        abortController = null;
        startGpuPolling(false);
    }
});

function showResult(data) {
    setTimeout(() => {
        els.processingSection.classList.add("hidden");
        els.resultSection.classList.remove("hidden");

        const hasSpeakers = data.segments?.some(s => s.speaker);
        els.resultText.textContent = hasSpeakers
            ? data.segments.map(s => (s.speaker ? `[${s.speaker}]  ` : "") + s.text.trim()).join("\n")
            : data.text;

        els.metaLang.textContent     = (data.language || "—").toUpperCase();
        els.metaDuration.textContent = fmtDuration(data.duration);
        els.metaTime.textContent     = `${data.processing_time_seconds}s`;
    }, 1200);
}

function showDownloads(files) {
    els.downloadBar.innerHTML = "";
    for (const [label, path] of [["TXT", files.txt], ["SRT", files.srt], ["JSON", files.json]]) {
        const a = document.createElement("a");
        a.href = path; a.download = ""; a.className = "dl-btn"; a.textContent = label;
        els.downloadBar.appendChild(a);
    }
    els.downloadBar.classList.remove("hidden");
}

els.copyBtn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(els.resultText.textContent);
    const orig = els.copyBtn.innerHTML;
    els.copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copiado!`;
    setTimeout(() => { els.copyBtn.innerHTML = orig; }, 2000);
});

els.cancelBtn.addEventListener("click", () => { if (abortController) abortController.abort(); });
els.newBtn.addEventListener("click", resetToUpload);

/* ── History ─────────────────────────────────────────────────── */
els.refreshHistoryBtn.addEventListener("click", loadHistory);

async function loadHistory() {
    els.historyList.innerHTML = `
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-card"></div>`;

    try {
        const items = await fetch("/api/history").then(r => r.json());
        if (!items.length) {
            els.historyList.innerHTML = `
                <div class="empty-state">
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/>
                    </svg>
                    <p>Nenhuma transcrição ainda</p>
                    <span>Transcreva um arquivo para ele aparecer aqui</span>
                </div>`;
            return;
        }
        els.historyList.innerHTML = items.map(renderHistoryCard).join("");
    } catch {
        els.historyList.innerHTML = `
            <div class="empty-state">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <p>Erro ao carregar histórico</p>
                <span>Verifique se a API está rodando</span>
            </div>`;
    }
}

function renderHistoryCard(item) {
    const date = new Date(item.created_at * 1000).toLocaleString("pt-BR", {
        day:"2-digit", month:"short", hour:"2-digit", minute:"2-digit"
    });
    const lang = item.language ? item.language.toUpperCase() : "—";
    const dur  = item.duration ? fmtDuration(item.duration) : "—";
    const dls  = Object.entries(item.files)
        .map(([ext, path]) => `<a href="${path}" download class="dl-btn">${ext.toUpperCase()}</a>`)
        .join("");

    return `
    <div class="history-card">
        <div class="history-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="23 7 16 12 23 17 23 7"/><rect width="15" height="14" x="1" y="5" rx="2" ry="2"/>
            </svg>
        </div>
        <div class="history-main">
            <div class="history-title">${esc(item.filename || item.folder)}</div>
            <div class="history-sub">${esc(item.folder)}</div>
        </div>
        <div class="history-meta">
            <span class="chip chip-purple">${lang}</span>
            <span class="chip">${dur}</span>
            <span class="chip">${date}</span>
        </div>
        <div class="history-actions">${dls}</div>
    </div>`;
}

function esc(s) {
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

/* ── Settings ────────────────────────────────────────────────── */
const SETTINGS_KEY = "wt_settings";

function getSettings() {
    try { return { model:"large-v3-turbo", language:"pt", speed:"2.5", diarize:"false",
                   ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; }
    catch { return { model:"large-v3-turbo", language:"pt", speed:"2.5", diarize:"false" }; }
}

function applySettingsToForm() {
    const s = getSettings();
    els.modelSelect.value    = s.model;
    els.languageSelect.value = s.language;
    els.speedInput.value     = s.speed;
    els.diarizeSelect.value  = s.diarize;
}

function loadSettingsForm() {
    const s = getSettings();
    $("#cfg-model").value    = s.model;
    $("#cfg-language").value = s.language;
    $("#cfg-speed").value    = s.speed;
    $("#cfg-diarize").value  = s.diarize;
}

document.getElementById("unloadModelBtn").addEventListener("click", async () => {
    const btn = document.getElementById("unloadModelBtn");
    const msg = document.getElementById("unloadMsg");
    btn.disabled = true;
    btn.textContent = "Liberando...";
    try {
        const data = await fetch("/unload", { method: "POST" }).then(r => r.json());
        msg.textContent = data.message;
        msg.classList.add("visible");
        updateGpu();
        setTimeout(() => msg.classList.remove("visible"), 3000);
    } catch {
        msg.textContent = "Erro ao comunicar com a API.";
        msg.classList.add("visible");
        setTimeout(() => msg.classList.remove("visible"), 3000);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg> Liberar VRAM`;
    }
});

els.saveSettingsBtn.addEventListener("click", () => {
    const s = {
        model:   $("#cfg-model").value,
        language:$("#cfg-language").value,
        speed:   $("#cfg-speed").value,
        diarize: $("#cfg-diarize").value,
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    applySettingsToForm();

    els.savedMsg.classList.add("visible");
    setTimeout(() => els.savedMsg.classList.remove("visible"), 2500);
});

/* ── GPU monitor ─────────────────────────────────────────────── */
const gpuEls = {
    barFill: $("#gpuBarFill"),
    value:   $("#gpuValue"),
    util:    $("#gpuUtil"),
    temp:    $("#gpuTemp"),
};

async function updateGpu() {
    try {
        const d = await fetch("/gpu").then(r => r.json());
        const pct   = Math.round((d.vram_used_mb / d.vram_total_mb) * 100) || 0;
        const usedG = (d.vram_used_mb / 1024).toFixed(1);
        const totG  = (d.vram_total_mb / 1024).toFixed(1);
        gpuEls.barFill.style.width = pct + "%";
        gpuEls.barFill.className   = `gpu-bar-fill${pct > 80 ? " high" : ""}`;
        gpuEls.value.textContent   = `${usedG}/${totG}`;
        gpuEls.util.textContent    = `GPU ${d.gpu_util_pct}%`;
        gpuEls.temp.textContent    = `${d.temp_c}°C`;
    } catch {}
}

let gpuInterval = null;
function startGpuPolling(fast = false) {
    if (gpuInterval) clearInterval(gpuInterval);
    gpuInterval = setInterval(updateGpu, fast ? 2000 : 10000);
}

/* ── Helpers ─────────────────────────────────────────────────── */
function fmtSize(b) {
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b/1024).toFixed(1) + " KB";
    if (b < 1073741824) return (b/1048576).toFixed(1) + " MB";
    return (b/1073741824).toFixed(2) + " GB";
}

function fmtDuration(sec) {
    const m = Math.floor(sec / 60), s = Math.round(sec % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/* ── Init ────────────────────────────────────────────────────── */
applySettingsToForm();
checkHealth();
setInterval(checkHealth, 15000);
updateGpu();
startGpuPolling(false);
