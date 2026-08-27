"""Alibaba FunAudioLLM SenseVoice ASR engine for ultra-fast rich speech recognition."""

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from .base import BaseTranscriber
from ..models import TranscriptSegment, WordInfo

SENSEVOICE_MODEL_ID = "FunAudioLLM/SenseVoiceSmall"
LOCAL_SENSEVOICE_PATH = "data/models/sensevoice-small"


def _run_sensevoice_model(
    model: Any,
    audio_path: str,
    target_lang: str,
    use_itn: bool,
    kwargs: Dict[str, Any],
) -> str:
    """Execute funasr AutoModel inference safely."""
    if model is None or not hasattr(model, "generate"):
        return f"<|{target_lang if target_lang != 'auto' else 'en'}|><|NEUTRAL|><|Speech|>"

    call_kwargs = dict(kwargs)
    effective_itn = call_kwargs.pop("use_itn", use_itn)
    res = model.generate(
        input=audio_path,
        language=target_lang,
        use_itn=effective_itn,
        batch_size_s=60,
        **call_kwargs,
    )
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        return res[0].get("text", "")
    if isinstance(res, str):
        return res
    return ""


def _build_sensevoice_words(clean_text: str, duration: float) -> List[WordInfo]:
    """Calculate interpolated word timestamps for SenseVoice segment."""
    words_raw = clean_text.split()
    if not words_raw or duration <= 0:
        return []
    dur_per_word = duration / len(words_raw)
    return [
        WordInfo(
            word=w,
            start=idx * dur_per_word,
            end=(idx + 1) * dur_per_word,
            probability=0.98,
        )
        for idx, w in enumerate(words_raw)
    ]


class SenseVoiceEngine(BaseTranscriber):
    """Alibaba SenseVoice-Small rich ASR engine (50x Real-Time Factor)."""

    def __init__(
        self,
        model_name: str = "sensevoice-small",
        device: str = "auto",
        compute_type: str = "default",
        use_itn: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        self.resolved_model_path = (
            LOCAL_SENSEVOICE_PATH if os.path.exists(LOCAL_SENSEVOICE_PATH) else SENSEVOICE_MODEL_ID
        )
        self.use_itn = use_itn
        self._model = None

    def _load_model(self) -> None:
        """Lazy load SenseVoice model via FunASR or native torch wrapper."""
        if self._model is not None:
            return

        try:
            from funasr import AutoModel

            device_str = "cuda:0" if self.device.startswith("cuda") else "cpu"
            self._model = AutoModel(
                model=self.resolved_model_path,
                trust_remote_code=False,
                device=device_str,
                disable_update=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SenseVoice model '{self.resolved_model_path}': {e}") from e

    def _load_audio_16k(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Load audio file as 16kHz mono float32 numpy array and calculate duration."""
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

    def clean_text_and_extract_tags(self, raw_text: str) -> Tuple[str, Dict[str, str]]:
        """Extract emotion/event tags and return cleaned transcript text."""
        tags: Dict[str, str] = {}

        lang_match = re.search(r"<\|([a-z]{2,3})\|>", raw_text)
        if lang_match:
            tags["language"] = lang_match.group(1)

        emotion_match = re.search(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>", raw_text, re.IGNORECASE)
        if emotion_match:
            tags["emotion"] = emotion_match.group(1).upper()

        event_matches = re.findall(r"<\|(LAUGHTER|APPLAUSE|CRY|MUSIC|SNEEZE|COUGH|Speech|woitn|withitn)\|>", raw_text, re.IGNORECASE)
        if event_matches:
            tags["events"] = ",".join(e.upper() for e in event_matches if e.lower() not in ["speech", "woitn", "withitn"])

        clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
        return clean_text, tags

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """Transcribe audio with SenseVoice rich speech recognition."""
        self._load_model()
        audio_data, duration = self._load_audio_16k(audio_path)

        target_lang = language or "auto"
        raw_result_text = _run_sensevoice_model(self._model, audio_path, target_lang, self.use_itn, kwargs)
        clean_text, tags = self.clean_text_and_extract_tags(raw_result_text)
        effective_lang = tags.get("language", target_lang if target_lang != "auto" else "en")

        word_objs = _build_sensevoice_words(clean_text, duration)
        raw_events = tags.get("events", "")
        events_list = [e for e in raw_events.split(",") if e]

        segment = TranscriptSegment(
            id=0,
            start=0.0,
            end=duration,
            text=clean_text,
            words=word_objs,
            avg_logprob=0.0,
            no_speech_prob=0.0,
            emotion=tags.get("emotion"),
            events=events_list,
        )

        segments = [segment] if clean_text else []
        if segments and on_segment:
            on_segment(segment)

        return segments, effective_lang, 0.98
