"""
Whisper.cpp GGML / GGUF High-Efficiency C++ Engine.
Ultra-fast execution on CPU / ARM / Apple Silicon / GPU using GGML.
"""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo

WHISPER_CPP_MODEL_MAP: Dict[str, str] = {
    "whispercpp-tiny": "tiny",
    "whispercpp-base": "base",
    "whispercpp-small": "small",
    "whispercpp-medium": "medium",
    "whispercpp-turbo": "large-v3-turbo",
    "whispercpp-large-v3": "large-v3",
}


def _build_whispercpp_words(text: str, start: float, end: float) -> List[WordInfo]:
    """Interpolate word intervals across segment duration."""
    words = text.split()
    if not words:
        return []
    span = max(0.01, end - start)
    word_dur = span / len(words)
    return [
        WordInfo(
            word=w,
            start=round(start + idx * word_dur, 3),
            end=round(start + (idx + 1) * word_dur, 3),
            probability=0.95,
        )
        for idx, w in enumerate(words)
    ]


class WhisperCppTranscriber(BaseTranscriber):
    """Whisper.cpp engine using pywhispercpp GGML runtime."""

    def __init__(
        self,
        model_name: str = "whispercpp-base",
        device: str = "auto",
        compute_type: str = "default",
        n_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        self.resolved_size = WHISPER_CPP_MODEL_MAP.get(model_name, model_name.replace("whispercpp-", ""))
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) // 2)
        self._model = None

    def _load_model(self) -> None:
        """Lazy load pywhispercpp GGML model."""
        if self._model is not None:
            return
        try:
            from pywhispercpp.model import Model

            self._model = Model(self.resolved_size, n_threads=self.n_threads)
        except Exception as e:
            raise RuntimeError(f"Failed to load whisper.cpp model ({self.resolved_size}): {e}") from e

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Transcribe audio file using whisper.cpp GGML runtime."""
        self._load_model()

        params: Dict[str, Any] = {}
        if language and language != "auto":
            params["language"] = language

        raw_segments = self._model.transcribe(audio_path, **params)
        segments: List[TranscriptSegment] = []

        for idx, seg in enumerate(raw_segments, start=1):
            text = seg.text.strip()
            if not text:
                continue
            start_s = round(seg.t0 / 100.0, 3)
            end_s = round(seg.t1 / 100.0, 3)
            words = _build_whispercpp_words(text, start_s, end_s)
            t_seg = TranscriptSegment(
                id=idx,
                start=start_s,
                end=end_s,
                text=text,
                words=words,
            )
            segments.append(t_seg)
            if on_segment:
                on_segment(t_seg)

        detected_lang = language if language and language != "auto" else "en"
        return segments, detected_lang, 0.95
