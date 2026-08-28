"""
NVIDIA Nemotron-Labs-Audex-2B Audio-LLM Speech Engine.
Supports 2B Unified Audio-Text LLM with Instruct Mode and Thinking Mode.
"""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo


def _build_audex_words(text: str, duration: float) -> List[WordInfo]:
    """Calculate evenly spaced word intervals."""
    words = text.split()
    if not words or duration <= 0:
        return []
    dur_per_word = duration / len(words)
    return [
        WordInfo(
            word=w,
            start=round(idx * dur_per_word, 3),
            end=round((idx + 1) * dur_per_word, 3),
            probability=0.97,
        )
        for idx, w in enumerate(words)
    ]


def _clean_audex_response(raw_text: str, thinking_mode: bool) -> str:
    """Filter or preserve reasoning tags based on thinking_mode."""
    if thinking_mode:
        return raw_text.strip()
    return re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()


class AudexTranscriber(BaseTranscriber):
    """NVIDIA Nemotron Audex-2B Audio-LLM transcriber."""

    def __init__(
        self,
        model_name: str = "nemotron-audex-2b",
        device: str = "auto",
        compute_type: str = "default",
        thinking_mode: bool = False,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        self.thinking_mode = thinking_mode
        self.temperature = temperature
        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        """Lazy load Audex-2B model and processor."""
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_id = "nvidia/Nemotron-Labs-Audex-2B"
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            dtype = torch.float16 if self.compute_type in ("float16", "int8_float16") else torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self._model = self._model.to("cpu")
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load NVIDIA Audex-2B model: {e}") from e

    def _get_audio_duration(self, audio_path: str) -> float:
        """Extract audio duration using soundfile."""
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
        """Execute Audex-2B unified audio-text inference."""
        self._load_model()
        duration = self._get_audio_duration(audio_path)

        prompt = (
            "<|audio|> Please transcribe the speech in the audio verbatim in its original spoken language without translating."
            if not self.thinking_mode
            else "<|audio|> Please analyze and transcribe the speech in the audio step-by-step in its original spoken language without translating."
        )

        inputs = self._processor(text=prompt, audios=[audio_path], return_tensors="pt")
        target_device = "cuda" if self.device == "cuda" else "cpu"
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        gen_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = self._processor.decode(gen_tokens, skip_special_tokens=True)
        clean_text = _clean_audex_response(raw_text, self.thinking_mode)

        words = _build_audex_words(clean_text, duration)
        segment = TranscriptSegment(
            id=1,
            start=0.0,
            end=round(duration, 3),
            text=clean_text,
            words=words,
        )

        if on_segment:
            on_segment(segment)

        return [segment], "en", 0.98
