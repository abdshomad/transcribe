"""Unit tests for Whisper.cpp GGML/GGUF high-efficiency engine."""

from unittest.mock import MagicMock, patch
import pytest
from transcribe.engines.base import BaseTranscriber
from transcribe.engines.factory import get_transcriber
from transcribe.engines.whisper_cpp import (
    WHISPER_CPP_MODEL_MAP,
    WhisperCppTranscriber,
    _build_whispercpp_words,
)


def test_whispercpp_words_helper():
    """Verify word timestamp interpolation."""
    words = _build_whispercpp_words("ask what you can do", 0.0, 2.5)
    assert len(words) == 5
    assert words[0].word == "ask"
    assert words[0].start == 0.0
    assert words[4].end == 2.5


def test_whispercpp_model_mappings():
    """Verify model map lookup."""
    assert WHISPER_CPP_MODEL_MAP["whispercpp-tiny"] == "tiny"
    assert WHISPER_CPP_MODEL_MAP["whispercpp-turbo"] == "large-v3-turbo"
    assert WHISPER_CPP_MODEL_MAP["whispercpp-large-v3"] == "large-v3"


def test_whispercpp_engine_init():
    """Verify transcriber initialization and properties."""
    engine = WhisperCppTranscriber(model_name="whispercpp-base", device="cpu", n_threads=4)
    assert isinstance(engine, BaseTranscriber)
    assert engine.model_name == "whispercpp-base"
    assert engine.resolved_size == "base"
    assert engine.n_threads == 4


def test_whispercpp_factory_resolution():
    """Verify factory instantiation for whisper.cpp."""
    engine = get_transcriber("whispercpp-base")
    assert isinstance(engine, WhisperCppTranscriber)


def test_whispercpp_mock_transcribe():
    """Verify mocked pywhispercpp transcription execution."""
    engine = WhisperCppTranscriber(model_name="whispercpp-tiny", device="cpu")
    mock_seg = MagicMock()
    mock_seg.t0 = 0
    mock_seg.t1 = 350
    mock_seg.text = " Hello GGML World "

    mock_model = MagicMock()
    mock_model.transcribe.return_value = [mock_seg]
    engine._model = mock_model

    emitted = []
    segs, lang, prob = engine.transcribe(
        "sample.wav",
        language="en",
        on_segment=lambda s: emitted.append(s),
    )

    assert len(segs) == 1
    assert segs[0].text == "Hello GGML World"
    assert segs[0].start == 0.0
    assert segs[0].end == 3.5
    assert len(segs[0].words) == 3
    assert len(emitted) == 1
    assert lang == "en"
    assert prob == 0.95
