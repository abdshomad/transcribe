"""Unit tests for NVIDIA Nemotron Audex-2B Audio-LLM Engine."""

from unittest.mock import MagicMock, patch
import pytest
from transcribe.engines.base import BaseTranscriber
from transcribe.engines.factory import get_transcriber
from transcribe.engines.audex_engine import (
    AudexTranscriber,
    _build_audex_words,
    _clean_audex_response,
)


def test_audex_words_helper():
    """Verify word timestamp intervals."""
    words = _build_audex_words("audex unified audio text llm", 3.0)
    assert len(words) == 5
    assert words[0].word == "audex"
    assert words[0].start == 0.0
    assert words[4].end == 3.0


def test_audex_clean_response():
    """Verify thinking tag stripping in instruct mode."""
    raw = "<think>Acoustic feature indicates male speech</think>Hello World"
    assert _clean_audex_response(raw, thinking_mode=False) == "Hello World"
    assert "<think>" in _clean_audex_response(raw, thinking_mode=True)


def test_audex_engine_init():
    """Verify transcriber initialization and properties."""
    engine = AudexTranscriber(model_name="nemotron-audex-2b", device="cpu", thinking_mode=False)
    assert isinstance(engine, BaseTranscriber)
    assert engine.model_name == "nemotron-audex-2b"
    assert not engine.thinking_mode


def test_audex_factory_resolution():
    """Verify factory instantiation for Audex-2B."""
    engine = get_transcriber("nemotron-audex-2b")
    assert isinstance(engine, AudexTranscriber)


def test_audex_mock_transcribe():
    """Verify mocked audio-llm generate inference flow."""
    engine = AudexTranscriber(model_name="nemotron-audex-2b", device="cpu")
    mock_processor = MagicMock()
    mock_processor.return_value = {"input_ids": MagicMock(shape=[1, 5])}
    mock_processor.decode.return_value = "Verbatim speech transcription test."

    mock_model = MagicMock()
    mock_model.generate.return_value = [MagicMock()]

    engine._processor = mock_processor
    engine._model = mock_model

    with patch.object(engine, "_get_audio_duration", return_value=2.5):
        emitted = []
        segs, lang, prob = engine.transcribe(
            "mock.wav",
            on_segment=lambda s: emitted.append(s),
        )

        assert len(segs) == 1
        assert segs[0].text == "Verbatim speech transcription test."
        assert segs[0].start == 0.0
        assert segs[0].end == 2.5
        assert len(segs[0].words) == 4
        assert len(emitted) == 1
        assert lang == "en"
        assert prob == 0.98
