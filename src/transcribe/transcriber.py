"""Whisper speech-to-text transcription engine using faster-whisper (backwards-compatibility module)."""

from .engines.faster_whisper import FasterWhisperEngine, MODEL_ALIASES
from .models import TranscriptSegment, WordInfo

# Backward compatibility alias
FasterWhisperTranscriber = FasterWhisperEngine

__all__ = [
    "FasterWhisperEngine",
    "FasterWhisperTranscriber",
    "MODEL_ALIASES",
    "TranscriptSegment",
    "WordInfo",
]
