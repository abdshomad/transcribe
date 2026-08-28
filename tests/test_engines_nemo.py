"""Unit tests for NVIDIA NeMo & Parakeet Speech Engine."""

from unittest.mock import MagicMock, patch
import pytest
from transcribe.engines.base import BaseTranscriber
from transcribe.engines.factory import get_transcriber
from transcribe.engines.nemo_engine import (
    NEMO_MODEL_MAP,
    SherpaNemoTranscriber,
    _build_nemo_words,
)


def test_nemo_words_helper():
    """Verify word timestamp interval calculation."""
    words = _build_nemo_words("nvidia parakeet fast conformer", 0.0, 2.0)
    assert len(words) == 4
    assert words[0].word == "nvidia"
    assert words[0].start == 0.0
    assert words[3].end == 2.0


def test_nemo_model_mappings():
    """Verify model alias mappings."""
    assert "parakeet-tdt-ctc-1.1b" in NEMO_MODEL_MAP["nvidia-parakeet-tdt-1.1b"]
    assert "parakeet-tdt" in NEMO_MODEL_MAP["parakeet-tdt-0.6b"]
    assert NEMO_MODEL_MAP["nvidia-nemotron-speech-3.5"] == "nvidia/nemotron-speech-asr"


def test_nemo_engine_init():
    """Verify transcriber initialization and properties."""
    engine = SherpaNemoTranscriber(model_name="nvidia-parakeet-tdt-1.1b", device="cpu")
    assert isinstance(engine, BaseTranscriber)
    assert engine.model_name == "nvidia-parakeet-tdt-1.1b"
    assert "parakeet" in engine.resolved_model_id


def test_nemo_factory_resolution():
    """Verify factory instantiation for NVIDIA NeMo."""
    engine = get_transcriber("nvidia-parakeet-tdt-1.1b")
    assert isinstance(engine, SherpaNemoTranscriber)


def test_nemo_mock_transcribe():
    """Verify mocked sherpa-onnx decode execution."""
    engine = SherpaNemoTranscriber(model_name="nvidia-parakeet-tdt-1.1b", device="cpu")
    mock_rec = MagicMock()
    mock_stream = MagicMock()
    mock_res = MagicMock()
    mock_res.text = "Hello NVIDIA Parakeet Speech"
    mock_stream.result = mock_res
    mock_rec.create_stream.return_value = mock_stream
    engine._recognizer = mock_rec

    with patch.object(engine, "_read_audio_16k", return_value=(MagicMock(), 3.0)):
        emitted = []
        segs, lang, prob = engine.transcribe(
            "sample.wav",
            on_segment=lambda s: emitted.append(s),
        )

        assert len(segs) == 1
        assert segs[0].text == "Hello NVIDIA Parakeet Speech"
        assert segs[0].start == 0.0
        assert segs[0].end == 3.0
        assert len(segs[0].words) == 4
        assert len(emitted) == 1
        assert lang == "en"
        assert prob == 0.97
