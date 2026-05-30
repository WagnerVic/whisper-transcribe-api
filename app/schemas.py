from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float

class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: Optional[List[WordTimestamp]] = None

class TranscriptionResponse(BaseModel):
    filename: str
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    text: str
    segments: List[TranscriptionSegment]
    processing_time_seconds: float
