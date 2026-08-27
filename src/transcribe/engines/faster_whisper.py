"""Faster-Whisper (CTranslate2) ASR engine implementation."""

import os
from typing import Any, Callable, List, Optional, Tuple
from faster_whisper import WhisperModel
from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo

MODEL_ALIASES = {
    "cahya-whisper-tiny-id": "data/models/cahya-whisper-tiny-id-ct2" if os.path.exists("data/models/cahya-whisper-tiny-id-ct2") else "cahya/whisper-tiny-id",
    "cahya-whisper-base-id": "data/models/cahya-whisper-base-id-ct2" if os.path.exists("data/models/cahya-whisper-base-id-ct2") else "cahya/whisper-base-id",
    "cahya-whisper-small-id": "data/models/cahya-whisper-small-id-ct2" if os.path.exists("data/models/cahya-whisper-small-id-ct2") else "cahya/whisper-small-id",
    "cahya-whisper-medium-id": "data/models/cahya-whisper-medium-id-ct2" if os.path.exists("data/models/cahya-whisper-medium-id-ct2") else "cahya/whisper-medium-id",
    "cahya-faster-whisper-medium-id": "data/models/cahya-whisper-medium-id-ct2" if os.path.exists("data/models/cahya-whisper-medium-id-ct2") else "cahya/whisper-medium-id",
}


class FasterWhisperEngine(BaseTranscriber):
    """Polymorphic faster-whisper CTranslate2 ASR engine."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        cpu_threads: int = 4,
        model_size: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        effective_name = model_size if model_size is not None else model_name
        super().__init__(model_name=effective_name, device=device, compute_type=compute_type, **kwargs)
        self.model_size = effective_name
        self.cpu_threads = cpu_threads

        if self.compute_type == "default":
            self.compute_type = "float16" if self.device == "cuda" else "int8"

        resolved_model = MODEL_ALIASES.get(effective_name, effective_name)
        self.model = WhisperModel(
            model_size_or_path=resolved_model,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Transcribe audio and extract word-level timing with live callback."""
        effective_lang = language
        if effective_lang is None:
            if "-id" in self.model_size or "indonesian" in self.model_size.lower():
                effective_lang = "id"
            elif self.model_size.endswith(".en"):
                effective_lang = "en"

        vad_params = kwargs.get("vad_parameters", dict(min_silence_duration_ms=500))

        segments_generator, info = self.model.transcribe(
            audio_path,
            language=effective_lang,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
        )

        segments: List[TranscriptSegment] = []
        for i, s in enumerate(segments_generator):
            word_list: List[WordInfo] = []
            if s.words:
                for w in s.words:
                    word_list.append(
                        WordInfo(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )

            seg = TranscriptSegment(
                id=i,
                start=s.start,
                end=s.end,
                text=s.text,
                words=word_list,
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
            )
            segments.append(seg)
            if on_segment:
                on_segment(seg)

        return segments, info.language, info.language_probability
