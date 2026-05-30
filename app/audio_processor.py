import subprocess
import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

class AudioProcessor:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def process_media(self, input_path: str, speed_up: float = 2.5) -> str:
        """
        Processa vídeo ou áudio via FFmpeg:
        1. Extrai o áudio (se for vídeo)
        2. Aplica speed-up para reduzir o tempo (ex: 2.5x)
        3. Reduz a qualidade/bitrate (64k, 16kHz) para o Whisper processar mais rápido
        Retorna o caminho do arquivo .wav processado
        """
        filename = Path(input_path).stem
        output_path = self.output_dir / f"{filename}_processed.wav"

        logger.info(f"Processando mídia {input_path} com speedup {speed_up}x")

        # No ffmpeg moderno, atempo suporta 0.5 a 100.0. 
        # A combinação -ar 16000 força o sample rate que o Whisper usa internamente, poupando CPU depois.
        # -ac 1 força mono
        
        command = [
            "ffmpeg",
            "-y", # Sobrescrever
            "-i", str(input_path),
            "-vn", # Remover vídeo
            "-af", f"atempo={speed_up}", # Acelerar áudio
            "-ar", "16000", # Sample rate 16kHz
            "-ac", "1", # Mono
            "-b:a", "64k", # Bitrate baixo
            str(output_path)
        ]

        try:
            # Roda o comando silenciosamente
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            logger.info(f"Áudio processado salvo em: {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no FFmpeg. Detalhes: {e}")
            raise RuntimeError("Falha ao processar arquivo via FFmpeg.")

    def cleanup(self, file_path: str):
        """Remove o arquivo temporário após a transcrição"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Arquivo temporário removido: {file_path}")
        except Exception as e:
            logger.warning(f"Não foi possível remover {file_path}: {e}")
