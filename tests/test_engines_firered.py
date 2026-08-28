"""Unit and integration tests for FireRedTranscriber engine."""

from unittest.mock import MagicMock, patch
import pytest
from transcribe.engines.base import BaseTranscriber
from transcribe.engines.factory import get_transcriber
from transcribe.engines.firered import (
    FireRedTranscriber,
    _build_firered_words,
    _resolve_firered_type,
)
from transcribe.models import TranscriptSegment


def test_resolve_firered_type():
    """Verify resolution of ASR variant type."""
    assert _resolve_firered_type("fireredasr-aed-l") == "aed"
    assert _resolve_firered_type("firered-aed") == "aed"
    assert _resolve_firered_type("fireredasr-llm-l") == "llm"
    assert _resolve_firered_type("fireredaudio-9b") == "llm"


def test_build_firered_words():
    """Verify word timestamp estimation helper."""
    words = _build_firered_words("hello world test", 3.0)
    assert len(words) == 3
    assert words[0].word == "hello"
    assert words[0].start == 0.0
    assert words[0].end == 1.0
    assert words[1].word == "world"
    assert words[2].end == 3.0


def test_firered_transcriber_initialization():
    """Verify transcriber properties and inheritance."""
    t = FireRedTranscriber("fireredasr-aed-l", device="cpu")
    assert isinstance(t, BaseTranscriber)
    assert t.model_name == "fireredasr-aed-l"
    assert t.asr_type == "aed"
    assert t.device == "cpu"


def test_firered_factory_resolution():
    """Verify factory instantiation for FireRed variants."""
    engine = get_transcriber("fireredasr-aed-l")
    assert isinstance(engine, FireRedTranscriber)


def test_firered_mock_transcribe():
    """Verify mocked transcription flow."""
    t = FireRedTranscriber("fireredasr-aed-l", device="cpu")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = [{"text": "Hello FireRed World"}]
    t._model = mock_model

    with patch.object(t, "_ensure_16k_wav", return_value=("mock.wav", 2.5)):
        emitted_segs = []
        segs, lang, prob = t.transcribe(
            "dummy.wav",
            on_segment=lambda s: emitted_segs.append(s),
        )
        assert len(segs) == 1
        assert segs[0].text == "Hello FireRed World"
        assert segs[0].end == 2.5
        assert len(segs[0].words) == 3
        assert len(emitted_segs) == 1
        assert lang == "en"
        assert prob == 0.98
