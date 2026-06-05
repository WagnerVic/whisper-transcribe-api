import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.audio_processor import AudioProcessor, SUPPORTED_EXTENSIONS
from app.schemas import TranscriptionResponse
from app.transcriber import Transcriber
from app.diarizer import diarize as run_diarization, assign_speakers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SILENT_PATHS = {"/gpu", "/health"}


class _SilentPathsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f' {path} ' in msg for path in _SILENT_PATHS)


def _install_access_log_filter() -> None:
    log = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _SilentPathsFilter) for f in log.filters):
        log.addFilter(_SilentPathsFilter())

OUTPUT_DIR = Path("output")
SPEED_FACTOR = 2.5
MAX_FILE_MB = 2048

_current_transcriber: tuple[str, Transcriber] | None = None
_project_dir_lock = threading.Lock()
ALLOWED_MODELS = {"large-v3-turbo", "large-v3", "medium"}
DEFAULT_MODEL = "large-v3-turbo"


def get_transcriber(model_size: str = DEFAULT_MODEL) -> Transcriber:
    global _current_transcriber
    if model_size not in ALLOWED_MODELS:
        raise HTTPException(422, f"Modelo '{model_size}' inválido. Use: {sorted(ALLOWED_MODELS)}")
    if _current_transcriber and _current_transcriber[0] == model_size:
        return _current_transcriber[1]
    logger.info(f"Carregando modelo {model_size}...")
    _current_transcriber = (
        model_size,
        Transcriber(model_size=model_size, device="auto", compute_type="int8_float16"),
    )
    logger.info(f"Modelo {model_size} carregado.")
    return _current_transcriber[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _install_access_log_filter()
    logger.info("API pronta.")
    yield


app = FastAPI(
    title="Whisper Transcribe API",
    description="API de transcrição de vídeos e áudios usando Faster-Whisper com aceleração GPU.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/api/history")
def get_history():
    if not OUTPUT_DIR.exists():
        return []
    items = []
    for folder in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        files = {}
        for ext in ("txt", "srt", "json"):
            matches = list(folder.glob(f"*.{ext}"))
            if matches:
                files[ext] = f"/output/{folder.name}/{matches[0].name}"
        if not files:
            continue
        meta = {"filename": folder.name, "language": None, "duration": None, "processing_time_seconds": None}
        json_files = list(folder.glob("*.json"))
        if json_files:
            try:
                data = json.loads(json_files[0].read_text(encoding="utf-8"))
                meta.update({k: data.get(k) for k in ("filename", "language", "duration", "processing_time_seconds")})
            except Exception:
                pass
        items.append({"folder": folder.name, "created_at": folder.stat().st_mtime, "files": files, **meta})
    return items


def _next_project_dir(project_name: str) -> Path:
    with _project_dir_lock:
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        existing = sorted(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else []
        max_num = 0
        for d in existing:
            if d.is_dir() and d.name[:3].isdigit():
                max_num = max(max_num, int(d.name[:3]))
        next_num = max_num + 1
        slug = project_name.strip().replace(" ", "-").lower()
        folder = OUTPUT_DIR / f"{next_num:03d}_{slug}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder


@app.get("/health")
def health_check():
    loaded = [_current_transcriber[0]] if _current_transcriber else []
    device = _current_transcriber[1].device if _current_transcriber else "cuda"
    return {
        "status": "ok",
        "device": device,
        "default_model": DEFAULT_MODEL,
        "loaded_models": loaded,
    }


@app.post("/unload")
def unload_model():
    global _current_transcriber
    if _current_transcriber is None:
        return {"unloaded": False, "message": "Nenhum modelo carregado."}
    model_name = _current_transcriber[0]
    _current_transcriber = None
    logger.info(f"Modelo {model_name} descarregado da VRAM.")
    return {"unloaded": True, "message": f"Modelo '{model_name}' removido da VRAM."}


@app.get("/gpu")
def gpu_status():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        parts = [x.strip() for x in result.stdout.strip().split(",")]
        return {
            "vram_used_mb": int(parts[0]),
            "vram_total_mb": int(parts[1]),
            "gpu_util_pct": int(parts[2]),
            "temp_c": int(parts[3]),
        }
    except Exception:
        return {"vram_used_mb": 0, "vram_total_mb": 0, "gpu_util_pct": 0, "temp_c": 0}


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File(description="Vídeo ou áudio para transcrever")],
    language: Annotated[
        Optional[str],
        Form(description="Código do idioma (ex: pt, en). Deixe vazio para detecção automática."),
    ] = None,
    speed_up: Annotated[
        float,
        Form(description="Fator de aceleração do áudio via FFmpeg antes da transcrição (padrão: 2.5x)"),
    ] = SPEED_FACTOR,
    stream: Annotated[
        bool,
        Form(description="Se true, retorna SSE com progresso em tempo real."),
    ] = False,
    project_name: Annotated[
        Optional[str],
        Form(description="Nome do projeto/pasta onde os arquivos serão salvos."),
    ] = None,
    model_size: Annotated[
        str,
        Form(description="Tamanho do modelo Whisper (large-v3-turbo, large-v3, medium)."),
    ] = DEFAULT_MODEL,
    diarize: Annotated[
        bool,
        Form(description="Se true, identifica os falantes via pyannote."),
    ] = False,
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Formato '{suffix}' não suportado. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"Arquivo excede o limite de {MAX_FILE_MB} MB.")

    original_filename = file.filename

    if not stream:
        with tempfile.TemporaryDirectory() as tmp_dir:
            uploaded_path = Path(tmp_dir) / original_filename
            uploaded_path.write_bytes(file_bytes)

            processor = AudioProcessor(output_dir=tmp_dir)
            media_result = processor.process_media(str(uploaded_path), speed_up=speed_up)

            transcriber = get_transcriber(model_size)
            result = transcriber.transcribe(
                audio_path=media_result["path"],
                original_filename=original_filename,
                language=language or None,
                speed_factor=speed_up,
            )

            if diarize:
                d_segs = run_diarization(media_result["path"])
                assign_speakers(result.segments, d_segs, speed_factor=speed_up)

        slug = project_name.strip().replace(" ", "-").lower() if project_name else None
        proj_dir = _next_project_dir(project_name or Path(original_filename).stem)
        file_name = slug or proj_dir.name.split("_", 1)[-1]
        transcriber.export_txt(result, str(proj_dir), export_name=file_name)
        transcriber.export_srt(result, str(proj_dir), export_name=file_name)
        transcriber.export_json(result, str(proj_dir), export_name=file_name)
        return result

    # --- Modo SSE ---
    async def event_stream():
        tmp_dir = tempfile.mkdtemp()
        stop_event = threading.Event()
        try:
            yield _sse_event("log", {"step": "upload", "message": f"📁 Arquivo recebido: {original_filename}", "progress": 5})

            uploaded_path = Path(tmp_dir) / original_filename
            uploaded_path.write_bytes(file_bytes)
            file_size_mb = len(file_bytes) / (1024 * 1024)
            yield _sse_event("log", {"step": "save", "message": f"💾 Salvo no disco ({file_size_mb:.1f} MB)", "progress": 10})
            await asyncio.sleep(0)

            yield _sse_event("log", {"step": "ffmpeg", "message": f"⚡ Processando áudio via FFmpeg (speed-up {speed_up}x)...", "progress": 15})
            await asyncio.sleep(0)

            processor = AudioProcessor(output_dir=tmp_dir)
            media_result = await asyncio.to_thread(
                processor.process_media, str(uploaded_path), speed_up
            )

            proc_dur = _fmt_time(media_result["processed_duration"])
            proc_size = media_result["processed_size"] / (1024 * 1024)
            yield _sse_event("log", {"step": "ffmpeg_done", "message": f"✅ Áudio processado: {proc_dur} — {proc_size:.1f} MB", "progress": 30})
            await asyncio.sleep(0)

            processed_path = media_result["path"]

            needs_load = not (_current_transcriber and _current_transcriber[0] == model_size)
            if needs_load:
                yield _sse_event("log", {"step": "model_load", "message": f"⏳ Carregando modelo {model_size} na GPU...", "progress": 31})
                await asyncio.sleep(0)
                transcriber = await asyncio.to_thread(get_transcriber, model_size)
                yield _sse_event("log", {"step": "model_load_done", "message": f"✅ Modelo {model_size} pronto", "progress": 34})
                await asyncio.sleep(0)
            else:
                transcriber = get_transcriber(model_size)
                yield _sse_event("log", {"step": "model_cached", "message": f"⚡ Modelo {model_size} já estava na GPU", "progress": 31})
                await asyncio.sleep(0)

            yield _sse_event("log", {"step": "whisper", "message": f"🤖 Transcrevendo com modelo {model_size} (GPU)...", "progress": 34})
            await asyncio.sleep(0)

            progress_queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def on_progress(current_sec, total_sec, percent, segment_text):
                progress_val = 34 + int(percent * 51)
                cur_fmt = _fmt_time(current_sec)
                tot_fmt = _fmt_time(total_sec)
                pct_int = int(percent * 100)
                preview = segment_text[:80] + ("..." if len(segment_text) > 80 else "")
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {
                        "step": "whisper_progress",
                        "message": f"🎙️ {cur_fmt} / {tot_fmt} ({pct_int}%) — \"{preview}\"",
                        "progress": progress_val,
                    }
                )

            task = asyncio.ensure_future(
                asyncio.to_thread(
                    transcriber.transcribe,
                    processed_path,
                    original_filename,
                    language or None,
                    speed_up,
                    on_progress,
                    stop_event,
                )
            )

            while not task.done():
                if await request.is_disconnected():
                    logger.info("Cliente desconectou — cancelando transcrição.")
                    stop_event.set()
                    task.cancel()
                    return
                try:
                    msg = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                    yield _sse_event("log", msg)
                except asyncio.TimeoutError:
                    pass

            while not progress_queue.empty():
                msg = progress_queue.get_nowait()
                yield _sse_event("log", msg)

            result = await task

            yield _sse_event("log", {"step": "whisper_done", "message": f"✅ Transcrição concluída em {result.processing_time_seconds}s", "progress": 85})
            await asyncio.sleep(0)

            if diarize:
                yield _sse_event("log", {"step": "diarize", "message": "👥 Identificando falantes (pyannote)...", "progress": 86})
                await asyncio.sleep(0)

                d_queue = asyncio.Queue()
                d_loop = asyncio.get_running_loop()

                def on_diarize_progress(step_name, completed, total):
                    pct = int((completed / total) * 100) if total else 0
                    d_loop.call_soon_threadsafe(
                        d_queue.put_nowait,
                        {
                            "step": "diarize_progress",
                            "message": f"👥 Diarização ({step_name}): {completed}/{total} ({pct}%)",
                            "progress": 86 + int((completed / total) * 2),
                        }
                    )

                d_task = asyncio.ensure_future(
                    asyncio.to_thread(run_diarization, media_result["path"], on_diarize_progress)
                )

                while not d_task.done():
                    if await request.is_disconnected():
                        logger.info("Cliente desconectou — cancelando diarização.")
                        stop_event.set()
                        d_task.cancel()
                        return
                    try:
                        msg = await asyncio.wait_for(d_queue.get(), timeout=0.5)
                        yield _sse_event("log", msg)
                    except asyncio.TimeoutError:
                        pass

                while not d_queue.empty():
                    msg = d_queue.get_nowait()
                    yield _sse_event("log", msg)

                d_segs = await d_task
                assign_speakers(result.segments, d_segs, speed_factor=speed_up)
                yield _sse_event("log", {"step": "diarize_done", "message": "✅ Falantes identificados!", "progress": 89})
                await asyncio.sleep(0)

            slug = project_name.strip().replace(" ", "-").lower() if project_name else None
            proj_dir = _next_project_dir(project_name or Path(original_filename).stem)
            file_name = slug or proj_dir.name.split("_", 1)[-1]
            yield _sse_event("log", {"step": "export", "message": f"📝 Exportando para {proj_dir.name}/...", "progress": 90})
            await asyncio.sleep(0)

            transcriber.export_txt(result, str(proj_dir), export_name=file_name)
            transcriber.export_srt(result, str(proj_dir), export_name=file_name)
            transcriber.export_json(result, str(proj_dir), export_name=file_name)
            yield _sse_event("log", {"step": "export_done", "message": f"✅ Arquivos salvos em output/{proj_dir.name}/", "progress": 95})

            yield _sse_event("result", result.model_dump())
            yield _sse_event("files", {
                "folder": proj_dir.name,
                "txt": f"/output/{proj_dir.name}/{file_name}.txt",
                "srt": f"/output/{proj_dir.name}/{file_name}.srt",
                "json": f"/output/{proj_dir.name}/{file_name}.json",
            })
            yield _sse_event("log", {"step": "done", "message": "🎉 Tudo pronto!", "progress": 100})

            global _current_transcriber
            _current_transcriber = None
            logger.info("Modelo descarregado automaticamente após transcrição.")

        except Exception as e:
            yield _sse_event("error", {"message": str(e)})
        finally:
            stop_event.set()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
