import subprocess
import json
import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def probe_duration(file_path: str) -> float:
    """Retorna a duração do arquivo em segundos via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(file_path),
            ],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        return float(data["format"].get("duration", 0))
    except Exception:
        return 0.0


class AudioProcessor:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def process_media(self, input_path: str, speed_up: float = 2.5) -> dict:
        """
        Processa vídeo ou áudio via FFmpeg:
        1. Extrai o áudio (se for vídeo)
        2. Aplica speed-up para reduzir o tempo (ex: 2.5x)
        3. Reduz a qualidade/bitrate (64k, 16kHz) para o Whisper processar mais rápido
        Retorna dict com path e metadados
        """
        filename = Path(input_path).stem
        input_size = os.path.getsize(input_path)
        input_duration = probe_duration(input_path)

        # Etapa 1: Extrair áudio bruto (WAV mono 16kHz, sem speed-up)
        extracted_path = self.output_dir / f"{filename}_extracted.wav"
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            str(extracted_path),
        ]

        try:
            subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro na extração FFmpeg: {e}")
            raise RuntimeError("Falha ao extrair áudio via FFmpeg.")

        extracted_size = os.path.getsize(extracted_path)
        extracted_duration = probe_duration(str(extracted_path))

        # Etapa 2: Acelerar + reduzir bitrate
        processed_path = self.output_dir / f"{filename}_processed.wav"
        speed_cmd = [
            "ffmpeg", "-y",
            "-i", str(extracted_path),
            "-af", f"atempo={speed_up}",
            "-b:a", "64k",
            str(processed_path),
        ]

        try:
            subprocess.run(speed_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no speed-up FFmpeg: {e}")
            raise RuntimeError("Falha ao acelerar áudio via FFmpeg.")

        processed_size = os.path.getsize(processed_path)
        processed_duration = probe_duration(str(processed_path))

        # Limpa o intermediário
        extracted_path.unlink(missing_ok=True)

        return {
            "path": str(processed_path),
            "input_size": input_size,
            "input_duration": input_duration,
            "extracted_size": extracted_size,
            "extracted_duration": extracted_duration,
            "processed_size": processed_size,
            "processed_duration": processed_duration,
        }

    def cleanup(self, file_path: str):
        """Remove o arquivo temporário após a transcrição"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Arquivo temporário removido: {file_path}")
        except Exception as e:
            logger.warning(f"Não foi possível remover {file_path}: {e}")
