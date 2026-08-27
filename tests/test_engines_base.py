"""Tests for BaseTranscriber abstract interface and device resolution."""

import pytest
from typing import Callable, List, Optional, Tuple
from transcribe.engines.base import BaseTranscriber
from transcribe.models import TranscriptSegment, WordInfo


class DummyTranscriber(BaseTranscriber):
    """Concrete implementation for testing base class."""

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        seg = TranscriptSegment(
            id=0,
            start=0.0,
            end=1.5,
            text="Halo dunia",
            words=[WordInfo(word="Halo", start=0.0, end=0.7), WordInfo(word="dunia", start=0.8, end=1.5)],
        )
        if on_segment:
            on_segment(seg)
        return [seg], language or "id", 0.99


def test_base_transcriber_instantiation_and_properties():
    """Test BaseTranscriber initializes properties and handles device resolution."""
    engine = DummyTranscriber(model_name="test-model", device="cpu", compute_type="float32", custom_opt=123)
    assert engine.model_name == "test-model"
    assert engine.device == "cpu"
    assert engine.compute_type == "float32"
    assert engine.extra_kwargs.get("custom_opt") == 123


def test_base_transcriber_cannot_instantiate_abstract():
    """Test that BaseTranscriber cannot be instantiated directly without transcribe()."""
    with pytest.raises(TypeError):
        BaseTranscriber(model_name="dummy")  # type: ignore


def test_dummy_transcriber_transcribe_and_callback():
    """Test standard transcribe signature and callback emission."""
    engine = DummyTranscriber(model_name="dummy")
    collected = []

    segments, lang, prob = engine.transcribe(
        audio_path="dummy.wav",
        language="id",
        on_segment=lambda s: collected.append(s),
    )

    assert len(segments) == 1
    assert segments[0].text == "Halo dunia"
    assert lang == "id"
    assert prob == 0.99
    assert len(collected) == 1
    assert collected[0].text == "Halo dunia"


def test_base_transcriber_resolve_device():
    """Test resolve_device helper."""
    assert BaseTranscriber.resolve_device("cpu") == "cpu"
    resolved = BaseTranscriber.resolve_device("auto")
    assert resolved in ["cpu", "cuda"]
