"""Abstract base class and contract for all ASR transcription engines."""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple
from ..models import TranscriptSegment


class BaseTranscriber(ABC):
    """Polymorphic base interface for ASR engines."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "default",
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = self.resolve_device(device)
        self.compute_type = compute_type
        self.extra_kwargs = kwargs

    @staticmethod
    def resolve_device(device: str = "auto") -> str:
        """Resolve 'auto' device string to 'cuda' or 'cpu'."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """
        Execute speech recognition on audio file.

        Args:
            audio_path: Path to audio file (16kHz WAV or supported audio).
            language: Target ISO-639 language code or None for auto-detect.
            beam_size: Beam search width (if supported by engine).
            vad_filter: Whether to apply Voice Activity Detection.
            on_segment: Callback invoked on each emitted segment.
            **kwargs: Engine-specific transcription parameters.

        Returns:
            Tuple containing:
                - List of TranscriptSegment objects.
                - Detected or assigned language code string.
                - Language detection confidence probability (0.0 to 1.0).
        """
        raise NotImplementedError("Subclasses must implement transcribe()")
