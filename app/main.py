import asyncio
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.audio_processor import AudioProcessor, SUPPORTED_EXTENSIONS
from app.schemas import TranscriptionResponse
from app.transcriber import Transcriber
from app.diarizer import diarize as run_diarization, assign_speakers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper Transcribe API",
    description="API de transcrição de vídeos e áudios usando Faster-Whisper com aceleração GPU.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

OUTPUT_DIR = Path("output")
SPEED_FACTOR = 2.5


def _next_project_dir(project_name: str) -> Path:
    """Cria pasta numerada: 001_nome, 002_nome, ..."""
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

_transcribers: dict[str, Transcriber] = {}
ALLOWED_MODELS = {"large-v3-turbo", "large-v3", "medium"}
DEFAULT_MODEL = "large-v3-turbo"


def get_transcriber(model_size: str = DEFAULT_MODEL) -> Transcriber:
    if model_size not in ALLOWED_MODELS:
        model_size = DEFAULT_MODEL
    if model_size not in _transcribers:
        logger.info(f"Carregando modelo {model_size}...")
        _transcribers[model_size] = Transcriber(
            model_size=model_size,
            device="auto",
            compute_type="int8_float16",
        )
        logger.info(f"Modelo {model_size} carregado.")
    return _transcribers[model_size]


@app.on_event("startup")
async def startup_event():
    logger.info("Inicializando modelo Whisper padrão...")
    get_transcriber(DEFAULT_MODEL)
    logger.info("API pronta.")


@app.get("/health")
def health_check():
    transcriber = get_transcriber()
    loaded = list(_transcribers.keys())
    return {
        "status": "ok",
        "device": transcriber.device,
        "default_model": DEFAULT_MODEL,
        "loaded_models": loaded,
    }


@app.get("/gpu")
def gpu_status():
    import subprocess as sp
    try:
        result = sp.run(
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

    # Lê o arquivo na memória para poder usar no generator
    file_bytes = await file.read()
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
        try:
            # Etapa 1: Upload recebido
            yield _sse_event("log", {"step": "upload", "message": f"📁 Arquivo recebido: {original_filename}", "progress": 5})

            # Etapa 2: Salvando no disco
            uploaded_path = Path(tmp_dir) / original_filename
            uploaded_path.write_bytes(file_bytes)
            file_size_mb = len(file_bytes) / (1024 * 1024)
            yield _sse_event("log", {"step": "save", "message": f"💾 Salvo no disco ({file_size_mb:.1f} MB)", "progress": 10})
            await asyncio.sleep(0)

            # Etapa 3: FFmpeg
            yield _sse_event("log", {"step": "ffmpeg", "message": f"⚡ Processando áudio via FFmpeg (speed-up {speed_up}x)...", "progress": 15})
            await asyncio.sleep(0)

            processor = AudioProcessor(output_dir=tmp_dir)
            media_result = await asyncio.to_thread(
                processor.process_media, str(uploaded_path), speed_up
            )

            # Logs detalhados de tamanho e duração
            ext_dur = _fmt_time(media_result["extracted_duration"])
            ext_size = media_result["extracted_size"] / (1024 * 1024)
            yield _sse_event("log", {"step": "ffmpeg_extract", "message": f"🔊 Áudio extraído: {ext_dur} — {ext_size:.1f} MB", "progress": 22})

            proc_dur = _fmt_time(media_result["processed_duration"])
            proc_size = media_result["processed_size"] / (1024 * 1024)
            yield _sse_event("log", {"step": "ffmpeg_done", "message": f"✅ Após speed-up {speed_up}x: {proc_dur} — {proc_size:.1f} MB", "progress": 30})
            await asyncio.sleep(0)

            processed_path = media_result["path"]
            # Etapa 4: Transcrição com progresso granular
            yield _sse_event("log", {"step": "whisper", "message": f"🤖 Transcrevendo com modelo {model_size} (GPU)...", "progress": 30})
            await asyncio.sleep(0)

            progress_queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def on_progress(current_sec, total_sec, percent, segment_text):
                """Callback chamado a cada segmento — roda na thread do Whisper"""
                progress_val = 30 + int(percent * 55)  # 30% → 85%
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

            transcriber = get_transcriber(model_size)

            # Roda a transcrição em thread separada
            task = asyncio.ensure_future(
                asyncio.to_thread(
                    transcriber.transcribe,
                    processed_path,
                    original_filename,
                    language or None,
                    speed_up,
                    on_progress,
                )
            )

            # Consome a fila de progresso enquanto a transcrição roda
            while not task.done():
                try:
                    msg = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                    yield _sse_event("log", msg)
                except asyncio.TimeoutError:
                    pass

            # Drena mensagens restantes na fila
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
                d_loop = asyncio.get_event_loop()
                
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

            # Etapa 5: Exportação em pasta numerada
            slug = project_name.strip().replace(" ", "-").lower() if project_name else None
            proj_dir = _next_project_dir(project_name or Path(original_filename).stem)
            file_name = slug or proj_dir.name.split("_", 1)[-1]
            yield _sse_event("log", {"step": "export", "message": f"📝 Exportando para {proj_dir.name}/...", "progress": 90})
            await asyncio.sleep(0)

            transcriber.export_txt(result, str(proj_dir), export_name=file_name)
            transcriber.export_srt(result, str(proj_dir), export_name=file_name)
            transcriber.export_json(result, str(proj_dir), export_name=file_name)
            yield _sse_event("log", {"step": "export_done", "message": f"✅ Arquivos salvos em output/{proj_dir.name}/", "progress": 95})

            # Etapa 6: Resultado final
            yield _sse_event("result", result.model_dump())
            yield _sse_event("log", {"step": "done", "message": "🎉 Tudo pronto!", "progress": 100})

        except Exception as e:
            yield _sse_event("error", {"message": str(e)})
        finally:
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

# Monta o frontend no root por último
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
