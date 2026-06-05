import subprocess
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def probe_duration(file_path: str) -> float:
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
        Extrai e processa áudio via FFmpeg em uma única passagem.
        Aplica atempo apenas se speed_up != 1.0.
        Retorna dict com path e metadados.
        """
        filename = Path(input_path).stem
        input_size = os.path.getsize(input_path)
        input_duration = probe_duration(input_path)

        processed_path = self.output_dir / f"{filename}_processed.wav"

        af_args = ["-af", f"atempo={speed_up}"] if speed_up != 1.0 else []
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
        ] + af_args + [str(processed_path)]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no processamento FFmpeg: {e}")
            raise RuntimeError("Falha ao processar áudio via FFmpeg.")

        processed_size = os.path.getsize(processed_path)
        processed_duration = probe_duration(str(processed_path))

        return {
            "path": str(processed_path),
            "input_size": input_size,
            "input_duration": input_duration,
            "processed_size": processed_size,
            "processed_duration": processed_duration,
        }
