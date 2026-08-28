"""Hugging Face Transformers Acoustic CTC ASR Engine (Wav2Vec2 & Meta MMS)."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo

TRANSFORMERS_CTC_ALIASES: Dict[str, str] = {
    "indonesian-wav2vec2-regional": "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese",
    "indonesian-wav2vec2-large-xlsr": "cahya/wav2vec2-large-xlsr-indonesian",
    "meta-omnilingual-asr": "facebook/mms-1b-all",
    "meta-mms-1b-all": "facebook/mms-1b-all",
    "meta-mms-1b-fl102": "facebook/mms-1b-fl102",
    "meta-mms-300m": "facebook/mms-300m-1400",
    "omniasr-ctc-300m": "bezzam/omniasr-ctc-300m-v2",
    "omniasr-ctc-1b": "bezzam/omniasr-ctc-1b-v2",
}
CTC_MODEL_ALIASES = TRANSFORMERS_CTC_ALIASES

LOCAL_CTC_MODEL_DIRS: Dict[str, str] = {
    "indonesian-wav2vec2-regional": "data/models/indonesian-wav2vec2-regional",
    "indonesian-wav2vec2-large-xlsr": "data/models/indonesian-wav2vec2-large-xlsr",
    "meta-omnilingual-asr": "data/models/mms-1b-all",
    "meta-mms-1b-all": "data/models/mms-1b-all",
}

ISO639_TO_MMS: Dict[str, str] = {
    "id": "ind",
    "indonesian": "ind",
    "en": "eng",
    "english": "eng",
    "jv": "jav",
    "javanese": "jav",
    "su": "sun",
    "sundanese": "sun",
    "ms": "zlm",
    "malay": "zlm",
    "zh": "cmn",
    "chinese": "cmn",
    "es": "spa",
    "spanish": "spa",
    "ar": "ara",
    "arabic": "ara",
    "fr": "fra",
    "french": "fra",
    "de": "deu",
    "german": "deu",
    "ja": "jpn",
    "japanese": "jpn",
    "ko": "kor",
    "korean": "kor",
}


def normalize_mms_lang(lang: str) -> str:
    """Normalize language code to 3-letter ISO code for Meta MMS adapters."""
    clean = lang.lower().strip()
    return ISO639_TO_MMS.get(clean, clean)


def _resolve_effective_ctc_lang(
    model_name: str,
    requested_lang: Optional[str],
    target_lang: Optional[str],
) -> str:
    """Determine effective ISO language code for CTC model."""
    if requested_lang:
        return requested_lang
    if target_lang:
        return target_lang
    if "indonesian" in model_name or "wav2vec2" in model_name:
        return "id"
    return "en"


def _decode_ctc_chunk(
    model: Any,
    processor: Any,
    chunk: np.ndarray,
    device: str,
    sample_rate: int = 16000,
) -> str:
    """Run CTC acoustic forward pass and decode text."""
    import torch

    model_dtype = getattr(model, "dtype", None)
    target_dtype = model_dtype if isinstance(model_dtype, torch.dtype) else torch.float32

    if hasattr(processor, "feature_extractor"):
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(device, dtype=target_dtype)
        attention_mask = inputs.attention_mask.to(device) if getattr(inputs, "attention_mask", None) is not None else None
    else:
        input_values = torch.from_numpy(chunk).unsqueeze(0).to(device, dtype=target_dtype)
        attention_mask = None

    with torch.no_grad():
        out = model(input_values, attention_mask=attention_mask)
        logits = getattr(out, "logits", out)

    if isinstance(logits, torch.Tensor):
        predicted_ids = torch.argmax(logits, dim=-1)
    else:
        predicted_ids = torch.zeros((1, 10), dtype=torch.long)
    return processor.batch_decode(predicted_ids)[0].strip()


def _build_ctc_segment(
    seg_id: int,
    start_time: float,
    end_time: float,
    clean_text: str,
) -> TranscriptSegment:
    """Build a TranscriptSegment with interpolated word timestamps."""
    words_raw = clean_text.split()
    dur_per_word = (end_time - start_time) / len(words_raw) if words_raw else 0.0
    word_objs = [
        WordInfo(
            word=w,
            start=start_time + (idx * dur_per_word),
            end=start_time + ((idx + 1) * dur_per_word),
            probability=0.95,
        )
        for idx, w in enumerate(words_raw)
    ]
    return TranscriptSegment(
        id=seg_id,
        start=start_time,
        end=end_time,
        text=clean_text,
        words=word_objs,
        avg_logprob=0.0,
        no_speech_prob=0.0,
    )


def _load_omniasr_ctc_model(model_id: str) -> Tuple[Any, Any]:
    """Load Meta OmniASR model using remapped Wav2Vec2ForCTC architecture and LasrTokenizer."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import LasrTokenizer, Wav2Vec2Config, Wav2Vec2ForCTC

    local_path = snapshot_download(model_id)
    tokenizer = LasrTokenizer.from_pretrained(local_path)
    cfg = Wav2Vec2Config(
        vocab_size=10288,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        feat_extract_norm="layer",
        feat_proj_dropout=0.0,
        layer_norm_eps=1e-5,
    )
    model = Wav2Vec2ForCTC(cfg)
    sd = load_file(os.path.join(local_path, "model.safetensors"))
    remapped = {}
    for k, v in sd.items():
        nk = f"wav2vec2.{k[8:]}" if k.startswith("encoder.") else (f"lm_head.{k[9:]}" if k.startswith("ctc_head.") else k)
        remapped[nk] = v
    model.load_state_dict(remapped, strict=False)
    return model, tokenizer


class TransformersCTCEngine(BaseTranscriber):
    """Acoustic CTC ASR engine supporting Wav2Vec2, Meta MMS and Meta OmniASR architectures."""

    def __init__(
        self,
        model_name: str = "indonesian-wav2vec2-regional",
        device: str = "auto",
        compute_type: str = "default",
        target_lang: Optional[str] = None,
        chunk_length_s: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        local_dir = LOCAL_CTC_MODEL_DIRS.get(model_name)
        if local_dir and os.path.exists(local_dir):
            self.resolved_model_id = local_dir
        else:
            self.resolved_model_id = TRANSFORMERS_CTC_ALIASES.get(model_name, model_name)

        self.target_lang = target_lang
        self.chunk_length_s = chunk_length_s
        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        """Load Hugging Face CTC model and processor lazily."""
        if self._model is not None and self._processor is not None:
            return

        try:
            if "omniasr" in self.resolved_model_id.lower():
                self._model, self._processor = _load_omniasr_ctc_model(self.resolved_model_id)
            else:
                from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor
                try:
                    self._processor = Wav2Vec2Processor.from_pretrained(self.resolved_model_id)
                except Exception:
                    self._processor = AutoProcessor.from_pretrained(self.resolved_model_id)
                self._model = AutoModelForCTC.from_pretrained(self.resolved_model_id)

            if self.device == "cuda":
                import torch
                dtype = torch.float16 if self.compute_type in ("float16", "int8_float16") else torch.float32
                self._model = self._model.to("cuda", dtype=dtype)
            else:
                self._model = self._model.to("cpu")

            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load CTC model '{self.resolved_model_id}': {e}") from e

    def _set_mms_language(self, target_lang: str) -> None:
        """Set adapter language dynamically for Meta MMS models."""
        is_mms = "mms" in self.resolved_model_id.lower() or "mms" in self.model_name.lower() or "omnilingual" in self.model_name.lower()
        if not is_mms or not self._processor or not self._model:
            return
        mms_code = normalize_mms_lang(target_lang)
        try:
            if hasattr(self._processor, "tokenizer") and hasattr(self._processor.tokenizer, "set_target_lang"):
                self._processor.tokenizer.set_target_lang(mms_code)
            if hasattr(self._model, "load_adapter"):
                self._model.load_adapter(mms_code)
        except Exception:
            pass

    def _load_audio_16k(self, audio_path: str) -> np.ndarray:
        """Load audio file as 16kHz mono float32 numpy array."""
        import soundfile as sf

        speech, sample_rate = sf.read(audio_path)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        if sample_rate != 16000:
            import scipy.signal
            num_samples = int(len(speech) * 16000 / sample_rate)
            speech = scipy.signal.resample(speech, num_samples)
        return speech.astype(np.float32)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Execute CTC speech recognition with sliding chunk windowing."""
        self._load_model()
        assert self._model is not None and self._processor is not None

        effective_lang = _resolve_effective_ctc_lang(self.model_name, language, self.target_lang)
        self._set_mms_language(effective_lang)

        audio_data = self._load_audio_16k(audio_path)
        sample_rate = 16000
        chunk_samples = int(self.chunk_length_s * sample_rate)
        total_samples = len(audio_data)

        segments: List[TranscriptSegment] = []
        seg_id = 0

        for start_idx in range(0, total_samples, chunk_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            chunk = audio_data[start_idx:end_idx]

            if len(chunk) < int(0.1 * sample_rate):
                continue

            chunk_start = start_idx / sample_rate
            chunk_end = end_idx / sample_rate

            clean_text = _decode_ctc_chunk(self._model, self._processor, chunk, self.device, sample_rate)
            if not clean_text:
                continue

            segment = _build_ctc_segment(seg_id, chunk_start, chunk_end, clean_text)
            segments.append(segment)
            if on_segment:
                on_segment(segment)
            seg_id += 1

        return segments, effective_lang, 0.95
