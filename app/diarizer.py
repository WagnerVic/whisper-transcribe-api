"""
Módulo de diarização: identifica quem fala em cada trecho do áudio.
Usa pyannote.audio com modelo speaker-diarization-3.1.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_pipeline = None


def get_diarization_pipeline():
    """Carrega o pipeline de diarização (lazy, cached)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN não encontrado no .env")

    logger.info("Carregando modelo de diarização pyannote...")
    from pyannote.audio import Pipeline

    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )

    # Move para GPU se disponível
    import torch
    if torch.cuda.is_available():
        _pipeline.to(torch.device("cuda"))
        logger.info("Diarização rodando em CUDA.")
    else:
        logger.info("Diarização rodando em CPU.")

    return _pipeline


from typing import Optional, Callable

class CustomProgressHook:
    def __init__(self, on_progress: Callable[[str, int, int], None]):
        self.on_progress = on_progress

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, step_name: str, step_artifact: any, file: dict = None, total: int = None, completed: int = None):
        if completed is None:
            completed = total = 1
        self.on_progress(step_name, completed, total)


def diarize(audio_path: str, on_progress: Optional[Callable[[str, int, int], None]] = None) -> list[dict]:
    """
    Executa a diarização no áudio.
    Retorna lista de segmentos: [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}, ...]
    """
    pipeline = get_diarization_pipeline()
    
    if on_progress:
        with CustomProgressHook(on_progress) as hook:
            diarization = pipeline(audio_path, hook=hook)
    else:
        diarization = pipeline(audio_path)

    # Pyannote 4.x retorna DiarizeOutput, Pyannote 3.x retorna Annotation
    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    logger.info(f"Diarização: {len(segments)} segmentos, {len(set(s['speaker'] for s in segments))} falantes")
    return segments


def assign_speakers(
    transcription_segments: list,
    diarization_segments: list[dict],
    speed_factor: float = 1.0,
) -> list:
    """
    Cruza segmentos de transcrição (Whisper) com segmentos de diarização (pyannote).
    Atribui o speaker com maior sobreposição temporal a cada segmento de transcrição.
    """
    for t_seg in transcription_segments:
        t_start = t_seg.start
        t_end = t_seg.end
        best_speaker = "SPEAKER_??"
        best_overlap = 0.0

        for d_seg in diarization_segments:
            # Escala os tempos do pyannote de volta pro tempo original do áudio
            d_start_scaled = d_seg["start"] * speed_factor
            d_end_scaled = d_seg["end"] * speed_factor

            # Calcula sobreposição temporal
            overlap_start = max(t_start, d_start_scaled)
            overlap_end = min(t_end, d_end_scaled)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

        t_seg.speaker = best_speaker

    return transcription_segments
