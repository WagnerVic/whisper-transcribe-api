import threading
import time
import logging
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from rich.console import Console

from app.schemas import TranscriptionResponse, TranscriptionSegment, WordTimestamp

logger = logging.getLogger(__name__)
console = Console()


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
            import ctranslate2
            ctranslate2.get_supported_compute_types("cuda")
            return "cuda"
        except Exception:
            return "cpu"

    def transcribe(
        self,
        audio_path: str,
        original_filename: str,
        language: Optional[str] = None,
        speed_factor: float = 1.0,
        on_progress: Optional[callable] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> TranscriptionResponse:
        start_time = time.time()

        segments_iter, info = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,
            beam_size=5,
            condition_on_previous_text=False,
        )

        total_duration = info.duration
        pydantic_segments = []
        full_text_parts = []

        for seg in segments_iter:
            if stop_event and stop_event.is_set():
                logger.info("Transcrição interrompida via stop_event.")
                break

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

            if on_progress and total_duration > 0:
                pct = min(seg.end / total_duration, 1.0)
                on_progress(
                    current_sec=seg.end * speed_factor,
                    total_sec=total_duration * speed_factor,
                    percent=pct,
                    segment_text=seg.text.strip(),
                )

        elapsed = time.time() - start_time

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

    def export_txt(self, result: TranscriptionResponse, output_dir: str, export_name: str = None) -> str:
        stem = export_name or Path(result.filename).stem
        out = Path(output_dir) / f"{stem}.txt"
        
        # Reconstrói o texto se tiver diarização
        has_speaker = any(s.speaker for s in result.segments)
        if has_speaker:
            lines = []
            for seg in result.segments:
                speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
                lines.append(f"{speaker_prefix}{seg.text.strip()}")
            text_content = "\n".join(lines)
        else:
            text_content = result.text
            
        out.write_text(text_content, encoding="utf-8")
        return str(out)

    def export_srt(self, result: TranscriptionResponse, output_dir: str, export_name: str = None) -> str:
        stem = export_name or Path(result.filename).stem
        out = Path(output_dir) / f"{stem}.srt"
        lines = []
        for i, seg in enumerate(result.segments, start=1):
            lines.append(str(i))
            lines.append(f"{_fmt_srt(seg.start)} --> {_fmt_srt(seg.end)}")
            speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
            lines.append(f"{speaker_prefix}{seg.text.strip()}")
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
        return str(out)

    def export_json(self, result: TranscriptionResponse, output_dir: str, export_name: str = None) -> str:
        stem = export_name or Path(result.filename).stem
        out = Path(output_dir) / f"{stem}.json"
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return str(out)


def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
