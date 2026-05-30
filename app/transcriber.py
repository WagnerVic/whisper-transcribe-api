import time
import logging
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from app.schemas import TranscriptionResponse, TranscriptionSegment, WordTimestamp

logger = logging.getLogger(__name__)
console = Console()

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus"}

SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "int8_float16",
    ):
        resolved_device = self._resolve_device(device)
        resolved_compute = compute_type if resolved_device == "cuda" else "int8"

        console.print(
            f"[bold cyan]Carregando modelo[/bold cyan] [yellow]{model_size}[/yellow] "
            f"em [green]{resolved_device.upper()}[/green] "
            f"([dim]{resolved_compute}[/dim])"
        )

        self.model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=resolved_compute,
        )
        self.device = resolved_device

        console.print("[bold green]✓ Modelo carregado[/bold green]")

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device

        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def transcribe(
        self,
        audio_path: str,
        original_filename: str,
        language: Optional[str] = None,
        speed_factor: float = 1.0,
    ) -> TranscriptionResponse:
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Transcrevendo...", total=None)

            segments_iter, info = self.model.transcribe(
                audio_path,
                language=language,
                vad_filter=True,
                word_timestamps=True,
            )

            raw_segments = list(segments_iter)

        elapsed = time.time() - start_time

        pydantic_segments = []
        full_text_parts = []

        for seg in raw_segments:
            words = None
            if seg.words:
                words = [
                    WordTimestamp(
                        word=w.word,
                        start=round(w.start * speed_factor, 3),
                        end=round(w.end * speed_factor, 3),
                        probability=round(w.probability, 4),
                    )
                    for w in seg.words
                ]

            pydantic_segments.append(
                TranscriptionSegment(
                    id=seg.id,
                    start=round(seg.start * speed_factor, 3),
                    end=round(seg.end * speed_factor, 3),
                    text=seg.text.strip(),
                    words=words,
                )
            )
            full_text_parts.append(seg.text.strip())

        return TranscriptionResponse(
            filename=original_filename,
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration=round(info.duration * speed_factor, 2),
            duration_after_vad=round(info.duration_after_vad * speed_factor, 2),
            text=" ".join(full_text_parts),
            segments=pydantic_segments,
            processing_time_seconds=round(elapsed, 2),
        )

    def export_txt(self, result: TranscriptionResponse, output_dir: str) -> str:
        out = Path(output_dir) / f"{Path(result.filename).stem}.txt"
        out.write_text(result.text, encoding="utf-8")
        return str(out)

    def export_srt(self, result: TranscriptionResponse, output_dir: str) -> str:
        out = Path(output_dir) / f"{Path(result.filename).stem}.srt"
        lines = []
        for i, seg in enumerate(result.segments, start=1):
            lines.append(str(i))
            lines.append(f"{_fmt_srt(seg.start)} --> {_fmt_srt(seg.end)}")
            lines.append(seg.text)
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
        return str(out)

    def export_json(self, result: TranscriptionResponse, output_dir: str) -> str:
        out = Path(output_dir) / f"{Path(result.filename).stem}.json"
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return str(out)


def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
