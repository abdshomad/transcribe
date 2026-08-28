"""
FireRed Team ASR Engine Implementation.
Supports FireRedASR-AED-L (1.1B Conformer-AED) and FireRedASR-LLM-L (7.8B LLM ASR).
"""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo


def _build_firered_words(clean_text: str, duration: float) -> List[WordInfo]:
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
            probability=0.98,
        )
        for idx, w in enumerate(words_raw)
    ]


def _resolve_firered_type(model_name: str) -> str:
    """Resolve ASR variant type ('aed' or 'llm') from model name."""
    clean = model_name.lower().strip()
    if "llm" in clean or "9b" in clean:
        return "llm"
    return "aed"


def _parse_firered_output(results: Any) -> str:
    """Safely extract transcription string from FireRedAsr return format."""
    if not results or not isinstance(results, list):
        return ""
    first_item = results[0]
    if isinstance(first_item, dict):
        return first_item.get("text", "").strip()
    if isinstance(first_item, str):
        return first_item.strip()
    return ""


def _detect_firered_language(clean_text: str, explicit_lang: Optional[str]) -> str:
    """Detect or fallback language for FireRed output."""
    if explicit_lang:
        return explicit_lang
    return "zh" if re.search(r"[\u4e00-\u9fff]", clean_text) else "en"


class FireRedTranscriber(BaseTranscriber):
    """FireRed Team industrial ASR and Audio-LLM transcriber."""

    def __init__(
        self,
        model_name: str = "fireredasr-aed-l",
        device: str = "auto",
        compute_type: str = "default",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device=device, compute_type=compute_type, **kwargs)
        self.asr_type = _resolve_firered_type(model_name)
        self._model = None

    def _load_model(self) -> None:
        """Lazy loader for FireRedASR pretrained models."""
        if self._model is not None:
            return
        try:
            import argparse
            import torch

            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([argparse.Namespace])

            from fireredasr.models.fireredasr import FireRedAsr

            self._model = FireRedAsr.from_pretrained(self.asr_type)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FireRed model '{self.model_name}' (type: {self.asr_type}): {e}"
            ) from e

    def _ensure_16k_wav(self, audio_path: str) -> Tuple[str, float]:
        """Ensure audio is 16kHz mono WAV and calculate its duration."""
        import soundfile as sf

        speech, sample_rate = sf.read(audio_path)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        if sample_rate != 16000:
            import scipy.signal

            num_samples = int(len(speech) * 16000 / sample_rate)
            speech = scipy.signal.resample(speech, num_samples)

        duration = len(speech) / 16000.0
        if audio_path.endswith(".wav") and sample_rate == 16000:
            return audio_path, duration

        temp_wav = Path(audio_path).with_suffix(".16k.wav")
        sf.write(str(temp_wav), speech, 16000)
        return str(temp_wav), duration

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Execute FireRed ASR speech recognition."""
        self._load_model()
        wav_path, duration = self._ensure_16k_wav(audio_path)

        args_dict = {
            "use_gpu": self.device.startswith("cuda"),
            "beam_size": beam_size,
            "batch_size": 1,
            "nbest": 1,
        }
        for k, v in kwargs.items():
            if k not in ("language", "vad_filter", "on_segment", "model_name"):
                args_dict[k] = v

        utt_id = f"utt_{Path(audio_path).stem}"
        results = self._model.transcribe(
            batch_uttid=[utt_id],
            batch_wav_path=[wav_path],
            args=args_dict,
        )

        clean_text = _parse_firered_output(results)
        words = _build_firered_words(clean_text, duration)
        segment = TranscriptSegment(
            id=1,
            start=0.0,
            end=round(duration, 3),
            text=clean_text,
            words=words,
        )
        if on_segment:
            on_segment(segment)

        detected_lang = _detect_firered_language(clean_text, language)
        return [segment], detected_lang, 0.98
