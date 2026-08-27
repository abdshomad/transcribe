"""UsefulSensors Moonshine ONNX and Transformer edge ASR engine."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo

MOONSHINE_MODEL_ALIASES: Dict[str, str] = {
    "moonshine-tiny": "UsefulSensors/moonshine-tiny",
    "moonshine-base": "UsefulSensors/moonshine-base",
}

LOCAL_MOONSHINE_DIRS: Dict[str, str] = {
    "moonshine-tiny": "data/models/moonshine/tiny",
    "moonshine-base": "data/models/moonshine/base",
}


def _create_word_timestamps(clean_text: str, duration: float) -> List[WordInfo]:
    """Calculate interpolated word timestamps across total duration."""
    words_raw = clean_text.split()
    if not words_raw or duration <= 0:
        return []
    dur_per_word = duration / len(words_raw)
    return [
        WordInfo(
            word=w,
            start=idx * dur_per_word,
            end=(idx + 1) * dur_per_word,
            probability=0.97,
        )
        for idx, w in enumerate(words_raw)
    ]


def _run_torch_moonshine(model: Any, processor: Any, audio_data: np.ndarray, device: str) -> str:
    """Execute PyTorch Hugging Face Moonshine inference pipeline."""
    import torch

    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    model_dtype = getattr(model, "dtype", None)
    target_dtype = model_dtype if isinstance(model_dtype, torch.dtype) else None

    if getattr(inputs, "input_features", None) is not None:
        feat = inputs.input_features.to(device, dtype=target_dtype)
    elif getattr(inputs, "input_values", None) is not None:
        feat = inputs.input_values.to(device, dtype=target_dtype)
    else:
        feat = list(inputs.values())[0].to(device, dtype=target_dtype)

    with torch.no_grad():
        predicted_ids = model.generate(feat, max_new_tokens=256)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


class MoonshineEngine(BaseTranscriber):
    """Zero-overhead edge ASR engine for UsefulSensors Moonshine models."""

    def __init__(
        self,
        model_name: str = "moonshine-base",
        device: str = "auto",
        compute_type: str = "default",
        chunk_length_s: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        local_dir = LOCAL_MOONSHINE_DIRS.get(model_name)
        if local_dir and os.path.exists(local_dir):
            self.resolved_model_id = local_dir
        else:
            self.resolved_model_id = MOONSHINE_MODEL_ALIASES.get(model_name, model_name)

        self.chunk_length_s = chunk_length_s
        self._processor = None
        self._model = None
        self._onnx_session = None

    def _load_model(self) -> None:
        """Initialize Moonshine ONNX or Transformers pipeline lazily."""
        if self._model is not None or self._onnx_session is not None:
            return

        try:
            import onnxruntime as ort
            onnx_path = Path(self.resolved_model_id) / "model.onnx"
            if onnx_path.exists():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
                self._onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
                return
        except Exception:
            pass

        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.resolved_model_id)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(self.resolved_model_id)
            if self.device == "cuda":
                import torch
                self._model = self._model.to("cuda", dtype=torch.float16)
            else:
                self._model = self._model.to("cpu")
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Moonshine model '{self.resolved_model_id}': {e}") from e

    def _load_audio_16k(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Load audio file as 16kHz mono float32 numpy array."""
        import soundfile as sf

        speech, sample_rate = sf.read(audio_path)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        if sample_rate != 16000:
            import scipy.signal
            num_samples = int(len(speech) * 16000 / sample_rate)
            speech = scipy.signal.resample(speech, num_samples)
        duration = len(speech) / 16000.0
        return speech.astype(np.float32), duration

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Transcribe audio with Moonshine edge model."""
        self._load_model()
        audio_data, duration = self._load_audio_16k(audio_path)
        effective_lang = language or "en"
        text = ""

        if self._onnx_session is not None and hasattr(self._onnx_session, "generate"):
            try:
                text = self._onnx_session.generate(audio_data)
            except Exception:
                text = ""
        elif self._model is not None and self._processor is not None:
            text = _run_torch_moonshine(self._model, self._processor, audio_data, self.device)

        clean_text = text.strip()
        word_objs = _create_word_timestamps(clean_text, duration)

        segment = TranscriptSegment(
            id=0,
            start=0.0,
            end=duration,
            text=clean_text,
            words=word_objs,
            avg_logprob=0.0,
            no_speech_prob=0.0,
        )

        segments = [segment] if clean_text else []
        if segments and on_segment:
            on_segment(segment)

        return segments, effective_lang, 0.97
