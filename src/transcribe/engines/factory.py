"""Dynamic ASR engine factory and registry with lazy imports."""

import importlib
from typing import Any, Callable, Dict, List, Optional, Type, Union
from .base import BaseTranscriber

# Default map of model identifiers to their engine import paths
DEFAULT_ENGINE_MAPPINGS: Dict[str, str] = {
    # Whisper & Distil-Whisper & CT2
    "tiny": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "tiny.en": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "base": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "base.en": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "small": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "small.en": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "medium": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "medium.en": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "large-v1": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "large-v2": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "large-v3": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "large-v3-turbo": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "turbo": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "distil-small.en": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "distil-medium.en": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "distil-large-v2": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "distil-large-v3": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "cahya-whisper-tiny-id": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "cahya-whisper-base-id": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "cahya-whisper-small-id": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "cahya-whisper-medium-id": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    "cahya-faster-whisper-medium-id": "transcribe.engines.faster_whisper.FasterWhisperEngine",
    # Hugging Face Transformers Acoustic CTC
    "indonesian-wav2vec2-regional": "transcribe.engines.transformers_ctc.TransformersCTCEngine",
    "indonesian-wav2vec2-large-xlsr": "transcribe.engines.transformers_ctc.TransformersCTCEngine",
    "meta-omnilingual-asr": "transcribe.engines.transformers_ctc.TransformersCTCEngine",
    "meta-mms-1b-all": "transcribe.engines.transformers_ctc.TransformersCTCEngine",
    "meta-mms-1b-fl102": "transcribe.engines.transformers_ctc.TransformersCTCEngine",
    "meta-mms-300m": "transcribe.engines.transformers_ctc.TransformersCTCEngine",
    "omniasr-ctc-300m": "transcribe.engines.transformers_ctc.TransformersCTCEngine",
    "omniasr-ctc-1b": "transcribe.engines.transformers_ctc.TransformersCTCEngine",
    # Alibaba FunAudioLLM SenseVoice
    "sensevoice-small": "transcribe.engines.sensevoice.SenseVoiceEngine",
    # UsefulSensors Moonshine ONNX
    "moonshine-tiny": "transcribe.engines.moonshine.MoonshineEngine",
    "moonshine-base": "transcribe.engines.moonshine.MoonshineEngine",
    # FireRed Team ASR & Audio-LLM
    "fireredasr-aed-l": "transcribe.engines.firered.FireRedTranscriber",
    "fireredasr-llm-l": "transcribe.engines.firered.FireRedTranscriber",
    "fireredaudio-9b": "transcribe.engines.firered.FireRedTranscriber",
    # Tsinghua VoiceMem Cognitive Audio Perception
    "voicemem-normal": "transcribe.engines.voicemem_engine.VoiceMemTranscriber",
    "voicemem-realtime": "transcribe.engines.voicemem_engine.VoiceMemTranscriber",
    # Whisper.cpp GGML/GGUF Engine
    "whispercpp-tiny": "transcribe.engines.whisper_cpp.WhisperCppTranscriber",
    "whispercpp-base": "transcribe.engines.whisper_cpp.WhisperCppTranscriber",
    "whispercpp-small": "transcribe.engines.whisper_cpp.WhisperCppTranscriber",
    "whispercpp-medium": "transcribe.engines.whisper_cpp.WhisperCppTranscriber",
    "whispercpp-turbo": "transcribe.engines.whisper_cpp.WhisperCppTranscriber",
    "whispercpp-large-v3": "transcribe.engines.whisper_cpp.WhisperCppTranscriber",
    # NVIDIA NeMo & Parakeet Speech Engine
    "nvidia-parakeet-tdt-1.1b": "transcribe.engines.nemo_engine.SherpaNemoTranscriber",
    "parakeet-tdt-1.1b": "transcribe.engines.nemo_engine.SherpaNemoTranscriber",
    "nvidia-parakeet-tdt-0.6b": "transcribe.engines.nemo_engine.SherpaNemoTranscriber",
    "parakeet-tdt-0.6b": "transcribe.engines.nemo_engine.SherpaNemoTranscriber",
    "nvidia-parakeet-ctc-1.1b": "transcribe.engines.nemo_engine.SherpaNemoTranscriber",
    "nvidia-nemotron-speech-3.5": "transcribe.engines.nemo_engine.SherpaNemoTranscriber",
    "nemotron-speech-3.5": "transcribe.engines.nemo_engine.SherpaNemoTranscriber",
    # NVIDIA Nemotron-Labs-Audex-2B Audio-LLM
    "nemotron-audex-2b": "transcribe.engines.audex_engine.AudexTranscriber",
    "nvidia-audex-2b": "transcribe.engines.audex_engine.AudexTranscriber",
}


def _resolve_engine_class(target: Union[str, Type[BaseTranscriber]]) -> Type[BaseTranscriber]:
    """Dynamically import and return engine class from dotpath or return directly."""
    if isinstance(target, type) and issubclass(target, BaseTranscriber):
        return target
    if isinstance(target, str):
        module_path, class_name = target.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        if not (isinstance(cls, type) and issubclass(cls, BaseTranscriber)):
            raise TypeError(f"Target '{target}' is not a subclass of BaseTranscriber")
        return cls
    raise ValueError(f"Invalid engine target specification: {target}")


class EngineRegistry:
    """Registry and factory for polymorphic ASR transcriber engines."""

    def __init__(self) -> None:
        self._registry: Dict[str, Union[str, Type[BaseTranscriber]]] = dict(DEFAULT_ENGINE_MAPPINGS)

    def register(self, model_name: str, engine_target: Union[str, Type[BaseTranscriber]]) -> None:
        """Register a new engine target for a given model identifier."""
        self._registry[model_name.lower().strip()] = engine_target

    def get_engine_class(self, model_name: str) -> Type[BaseTranscriber]:
        """Resolve engine class for a given model identifier with heuristic fallback."""
        cleaned_name = model_name.lower().strip()
        if cleaned_name in self._registry:
            return _resolve_engine_class(self._registry[cleaned_name])

        # Heuristic fallbacks
        if "wav2vec" in cleaned_name or "mms-" in cleaned_name:
            return _resolve_engine_class("transcribe.engines.transformers_ctc.TransformersCTCEngine")
        if "sensevoice" in cleaned_name:
            return _resolve_engine_class("transcribe.engines.sensevoice.SenseVoiceEngine")
        if "moonshine" in cleaned_name:
            return _resolve_engine_class("transcribe.engines.moonshine.MoonshineEngine")
        if "nemo" in cleaned_name or "parakeet" in cleaned_name:
            return _resolve_engine_class("transcribe.engines.nemo.NeMoEngine")

        # Default fallback to Faster-Whisper
        return _resolve_engine_class("transcribe.engines.faster_whisper.FasterWhisperEngine")

    def get_transcriber(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        **kwargs: Any,
    ) -> BaseTranscriber:
        """Instantiate transcriber instance dynamically."""
        cls = self.get_engine_class(model_name)
        return cls(model_name=model_name, device=device, compute_type=compute_type, **kwargs)

    def list_supported_models(self) -> List[str]:
        """List all explicitly mapped model identifiers."""
        return sorted(list(self._registry.keys()))


# Global default factory singleton
default_registry = EngineRegistry()


def get_transcriber(
    model_name: str = "base",
    device: str = "auto",
    compute_type: str = "default",
    **kwargs: Any,
) -> BaseTranscriber:
    """Convenience functional interface for instantiating an ASR transcriber."""
    return default_registry.get_transcriber(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        **kwargs,
    )
