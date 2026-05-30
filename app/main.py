import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.audio_processor import AudioProcessor, SUPPORTED_EXTENSIONS
from app.schemas import TranscriptionResponse
from app.transcriber import Transcriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper Transcribe API",
    description="API de transcrição de vídeos e áudios usando Faster-Whisper com aceleração GPU.",
    version="0.1.0",
)

OUTPUT_DIR = "output"
SPEED_FACTOR = 2.5

_transcriber: Optional[Transcriber] = None


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber(
            model_size="large-v3",
            device="auto",
            compute_type="int8_float16",
        )
    return _transcriber


@app.on_event("startup")
async def startup_event():
    logger.info("Inicializando modelo Whisper...")
    get_transcriber()
    logger.info("API pronta.")


@app.get("/health")
def health_check():
    transcriber = get_transcriber()
    return {
        "status": "ok",
        "device": transcriber.device,
        "model": "large-v3",
    }


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
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Formato '{suffix}' não suportado. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        uploaded_path = Path(tmp_dir) / file.filename
        with uploaded_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        processor = AudioProcessor(output_dir=tmp_dir)
        processed_path = processor.process_media(str(uploaded_path), speed_up=speed_up)

        transcriber = get_transcriber()
        result = transcriber.transcribe(
            audio_path=processed_path,
            original_filename=file.filename,
            language=language or None,
            speed_factor=speed_up,
        )

    transcriber.export_txt(result, OUTPUT_DIR)
    transcriber.export_srt(result, OUTPUT_DIR)
    transcriber.export_json(result, OUTPUT_DIR)

    return result
