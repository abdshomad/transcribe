"""
NVIDIA NeMo & Sherpa-ONNX Speech Recognition Engine.
Supports NVIDIA Parakeet (TDT / CTC) and Nemotron Speech models.
"""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo

NEMO_MODEL_MAP: Dict[str, str] = {
    "nvidia-parakeet-tdt-1.1b": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-ctc-1.1b-en-36000",
    "parakeet-tdt-1.1b": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-ctc-1.1b-en-36000",
    "nvidia-parakeet-tdt-0.6b": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-en-36000",
    "parakeet-tdt-0.6b": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-en-36000",
    "nvidia-parakeet-ctc-1.1b": "csukuangfj/sherpa-onnx-nemo-parakeet-ctc-1.1b-en-36000",
    "parakeet-ctc-1.1b": "csukuangfj/sherpa-onnx-nemo-parakeet-ctc-1.1b-en-36000",
    "nvidia-parakeet-ctc-0.6b": "csukuangfj/sherpa-onnx-nemo-parakeet-ctc-0.6b-en-36000",
    "parakeet-ctc-0.6b": "csukuangfj/sherpa-onnx-nemo-parakeet-ctc-0.6b-en-36000",
    "nvidia-nemotron-speech-3.5": "nvidia/nemotron-speech-asr",
    "nemotron-speech-3.5": "nvidia/nemotron-speech-asr",
}


def _build_nemo_words(text: str, start: float, end: float) -> List[WordInfo]:
    """Calculate word timestamp intervals."""
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
            probability=0.96,
        )
        for idx, w in enumerate(words)
    ]


def _create_sherpa_recognizer(local_path: str, device: str) -> Any:
    """Build Sherpa-ONNX recognizer from downloaded model directory."""
    import sherpa_onnx

    model_files = list(Path(local_path).glob("*.onnx"))
    tokens_file = list(Path(local_path).glob("*tokens.txt"))
    provider = "cuda" if device == "cuda" else "cpu"
    tok_path = str(tokens_file[0]) if tokens_file else ""

    if len(model_files) >= 3:
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=tok_path,
            encoder=str(model_files[0]),
            decoder=str(model_files[1]),
            joiner=str(model_files[2]),
            num_threads=4,
            provider=provider,
        )

    model_path = str(model_files[0]) if model_files else ""
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=model_path,
        tokens=tok_path,
        num_threads=4,
        provider=provider,
    )


class SherpaNemoTranscriber(BaseTranscriber):
    """NVIDIA NeMo Parakeet & Nemotron transcriber using sherpa-onnx runtime."""

    def __init__(
        self,
        model_name: str = "nvidia-parakeet-tdt-1.1b",
        device: str = "auto",
        compute_type: str = "default",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        self.resolved_model_id = NEMO_MODEL_MAP.get(model_name, model_name)
        self._recognizer = None

    def _load_model(self) -> None:
        """Lazy load sherpa-onnx offline recognizer."""
        if self._recognizer is not None:
            return
        try:
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(self.resolved_model_id)
            self._recognizer = _create_sherpa_recognizer(local_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load NVIDIA NeMo model ({self.resolved_model_id}): {e}") from e

    def _read_audio_16k(self, audio_path: str) -> Tuple[Any, float]:
        """Load audio samples normalized to 16kHz."""
        import soundfile as sf
        import numpy as np

        speech, sr = sf.read(audio_path, dtype="float32")
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            import soxr
            speech = soxr.resample(speech, sr, 16000)
        duration = len(speech) / 16000.0
        return speech, duration

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Transcribe audio with NVIDIA NeMo Parakeet / Nemotron."""
        self._load_model()
        samples, duration = self._read_audio_16k(audio_path)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        self._recognizer.decode_stream(stream)
        result = stream.result

        clean_text = getattr(result, "text", str(result)).strip()
        words = _build_nemo_words(clean_text, 0.0, duration)

        segment = TranscriptSegment(
            id=1,
            start=0.0,
            end=round(duration, 3),
            text=clean_text,
            words=words,
        )

        if on_segment:
            on_segment(segment)

        return [segment], "en", 0.97
