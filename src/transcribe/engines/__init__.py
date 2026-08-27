"""Speech-to-text polymorphic engine package."""

from .base import BaseTranscriber
from .factory import EngineRegistry, default_registry, get_transcriber
from .faster_whisper import FasterWhisperEngine
from .transformers_ctc import TransformersCTCEngine
from .sensevoice import SenseVoiceEngine
from .moonshine import MoonshineEngine

__all__ = [
    "BaseTranscriber",
    "EngineRegistry",
    "FasterWhisperEngine",
    "MoonshineEngine",
    "SenseVoiceEngine",
    "TransformersCTCEngine",
    "default_registry",
    "get_transcriber",
]
