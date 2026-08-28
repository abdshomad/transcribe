"""
Tsinghua VoiceMem Dual-Brain Cognitive Voice Perception & ASR Engine.
Combines ASR transcription with SER Emotion Detection & Speaker Identification.
"""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo


def _build_voicemem_words(clean_text: str, duration: float) -> List[WordInfo]:
    """Calculate evenly spaced word-level timestamp intervals."""
    words_raw = clean_text.split()
    if not words_raw or duration <= 0:
        return []
    dur_per_word = duration / len(words_raw)
    return [
        WordInfo(
            word=w,
            start=round(idx * dur_per_word, 3),
            end=round((idx + 1) * dur_per_word, 3),
            probability=0.96,
        )
        for idx, w in enumerate(words_raw)
    ]


def _detect_voicemem_lang(text: str, explicit_lang: Optional[str]) -> str:
    """Detect language code or fallback."""
    if explicit_lang:
        return explicit_lang
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"\b(dan|yang|di|ini|itu|ke|dari|untuk|pada|adalah)\b", text, re.IGNORECASE):
        return "id"
    return "en"


class VoiceMemTranscriber(BaseTranscriber):
    """VoiceMem dual-brain cognitive voice perception transcriber."""

    def __init__(
        self,
        model_name: str = "voicemem-normal",
        device: str = "auto",
        compute_type: str = "default",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        self.mode = "realtime" if "realtime" in model_name else "normal"
        self._vm = None

    def _load_model(self) -> None:
        """Lazy loader for VoiceMem perception instance."""
        if self._vm is not None:
            return
        try:
            from voicemem import VoiceMem

            self._vm = VoiceMem(mode=self.mode)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VoiceMem engine ({self.model_name}): {e}") from e

    def _get_audio_duration(self, audio_path: str) -> float:
        """Read audio duration using soundfile."""
        import soundfile as sf

        info = sf.info(audio_path)
        return float(info.duration)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Execute VoiceMem ingestion and multimodal transcript extraction."""
        self._load_model()
        duration = self._get_audio_duration(audio_path)

        res = self._vm.ingest(audio=audio_path)

        clean_text = ""
        emotion = "neutral"
        scene = "indoor"
        speaker_id = "SPEAKER_00"

        if isinstance(res, dict):
            clean_text = res.get("text", res.get("transcription", "")).strip()
            emotion = res.get("emotion", emotion)
            scene = res.get("scene", scene)
            speaker_id = res.get("speaker", speaker_id)
        elif hasattr(res, "text"):
            clean_text = getattr(res, "text", "").strip()

        words = _build_voicemem_words(clean_text, duration)
        segment = TranscriptSegment(
            id=1,
            start=0.0,
            end=round(duration, 3),
            text=clean_text,
            words=words,
        )

        if on_segment:
            on_segment(segment)

        detected_lang = _detect_voicemem_lang(clean_text, language)
        return [segment], detected_lang, 0.96
