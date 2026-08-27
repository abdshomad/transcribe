from dotenv import load_dotenv
from pathlib import Path

# Load secrets and env
load_dotenv(Path(".secrets"))
load_dotenv(Path(".env"))

from .models import (
    WordInfo,
    TranscriptSegment,
    SpeakerSegment,
    DiarizedSegment,
    TranscriptionResult,
)
from .pipeline import AudioTranscriptionPipeline
from .metrics import normalize_text, calculate_wer, calculate_cer, calculate_rtf
from .youtube import is_youtube_url, fetch_youtube_transcript

__version__ = "0.2.0"
__all__ = [
    "WordInfo",
    "TranscriptSegment",
    "SpeakerSegment",
    "DiarizedSegment",
    "TranscriptionResult",
    "AudioTranscriptionPipeline",
    "normalize_text",
    "calculate_wer",
    "calculate_cer",
    "calculate_rtf",
    "is_youtube_url",
    "fetch_youtube_transcript",
]
