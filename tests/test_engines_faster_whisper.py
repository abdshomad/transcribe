"""Tests for FasterWhisperEngine polymorphic wrapper."""

import pytest
from unittest.mock import MagicMock, patch
from transcribe.engines.faster_whisper import FasterWhisperEngine
from transcribe.engines.factory import get_transcriber
from transcribe.transcriber import FasterWhisperTranscriber


@patch("transcribe.engines.faster_whisper.WhisperModel")
def test_faster_whisper_engine_init(mock_whisper):
    """Test engine initialization and parameters."""
    engine = FasterWhisperEngine(model_name="tiny", device="cpu", compute_type="int8")
    assert engine.model_name == "tiny"
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"
    mock_whisper.assert_called_once_with(
        model_size_or_path="tiny",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
    )


@patch("transcribe.engines.faster_whisper.WhisperModel")
def test_faster_whisper_engine_transcribe_mock(mock_whisper_cls):
    """Test transcribe flow with mocked faster_whisper generator."""
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance

    # Mock whisper segment and info
    mock_word = MagicMock()
    mock_word.word = "Halo"
    mock_word.start = 0.0
    mock_word.end = 0.5
    mock_word.probability = 0.98

    mock_seg = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end = 1.0
    mock_seg.text = " Halo"
    mock_seg.words = [mock_word]
    mock_seg.avg_logprob = -0.1
    mock_seg.no_speech_prob = 0.01

    mock_info = MagicMock()
    mock_info.language = "id"
    mock_info.language_probability = 0.99

    mock_instance.transcribe.return_value = ([mock_seg], mock_info)

    engine = FasterWhisperEngine(model_name="cahya-whisper-tiny-id", device="cpu")
    collected = []
    segments, lang, prob = engine.transcribe(
        audio_path="test.wav",
        on_segment=lambda s: collected.append(s),
    )

    assert len(segments) == 1
    assert segments[0].text == " Halo"
    assert len(segments[0].words) == 1
    assert segments[0].words[0].word == "Halo"
    assert lang == "id"
    assert prob == 0.99
    assert len(collected) == 1


@patch("transcribe.engines.faster_whisper.WhisperModel")
def test_factory_instantiates_faster_whisper_engine(mock_whisper_cls):
    """Test get_transcriber factory returns FasterWhisperEngine for standard whisper models."""
    engine = get_transcriber("tiny", device="cpu")
    assert isinstance(engine, FasterWhisperEngine)
    assert isinstance(engine, FasterWhisperTranscriber)
