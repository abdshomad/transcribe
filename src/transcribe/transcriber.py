"""Whisper speech-to-text transcription engine using faster-whisper."""

import os
from typing import Callable, List, Optional, Tuple
from faster_whisper import WhisperModel
from .models import TranscriptSegment, WordInfo


MODEL_ALIASES = {
    "cahya-whisper-tiny-id": "data/models/cahya-whisper-tiny-id-ct2" if os.path.exists("data/models/cahya-whisper-tiny-id-ct2") else "cahya/whisper-tiny-id",
    "cahya-whisper-base-id": "data/models/cahya-whisper-base-id-ct2" if os.path.exists("data/models/cahya-whisper-base-id-ct2") else "cahya/whisper-base-id",
    "cahya-whisper-small-id": "data/models/cahya-whisper-small-id-ct2" if os.path.exists("data/models/cahya-whisper-small-id-ct2") else "cahya/whisper-small-id",
    "cahya-whisper-medium-id": "data/models/cahya-whisper-medium-id-ct2" if os.path.exists("data/models/cahya-whisper-medium-id-ct2") else "cahya/whisper-medium-id",
    "cahya-faster-whisper-medium-id": "data/models/cahya-whisper-medium-id-ct2" if os.path.exists("data/models/cahya-whisper-medium-id-ct2") else "cahya/whisper-medium-id",
}


class FasterWhisperTranscriber:
    """Wrapper around faster-whisper for fast and accurate ASR."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        cpu_threads: int = 4,
    ):
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "int8"

        resolved_model = MODEL_ALIASES.get(model_size, model_size)
        self.device = device
        self.compute_type = compute_type
        self.model_size = model_size
        self.model = WhisperModel(
            model_size_or_path=resolved_model,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Transcribe audio and extract word-level timing with live callback."""
        effective_lang = language
        if effective_lang is None:
            if "-id" in self.model_size or "indonesian" in self.model_size.lower():
                effective_lang = "id"
            elif self.model_size.endswith(".en"):
                effective_lang = "en"

        segments_generator, info = self.model.transcribe(
            audio_path,
            language=effective_lang,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
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
