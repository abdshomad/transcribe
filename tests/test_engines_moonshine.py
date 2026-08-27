"""Tests for MoonshineEngine (UsefulSensors Moonshine ONNX/Transformers)."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from transcribe.engines.moonshine import MoonshineEngine, MOONSHINE_MODEL_ALIASES
from transcribe.engines.factory import get_transcriber


def test_moonshine_model_aliases():
    """Verify alias mapping for Moonshine tiny and base models."""
    assert "moonshine-tiny" in MOONSHINE_MODEL_ALIASES
    assert "moonshine-base" in MOONSHINE_MODEL_ALIASES
    assert MOONSHINE_MODEL_ALIASES["moonshine-tiny"] == "UsefulSensors/moonshine-tiny"


def test_moonshine_engine_init():
    """Test engine instantiation and model resolution."""
    engine = MoonshineEngine(model_name="moonshine-tiny", device="cpu")
    assert engine.model_name == "moonshine-tiny"
    assert "tiny" in engine.resolved_model_id
    assert engine.device == "cpu"


@patch("soundfile.read")
def test_moonshine_transcribe_with_mock_onnx(mock_sf_read):
    """Test transcribe flow with mock ONNX session."""
    mock_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds
    mock_sf_read.return_value = (mock_audio, 16000)

    engine = MoonshineEngine(model_name="moonshine-tiny", device="cpu")
    mock_onnx = MagicMock()
    mock_onnx.generate.return_value = "edge speech recognition works seamlessly"
    engine._onnx_session = mock_onnx

    collected = []
    segments, lang, prob = engine.transcribe(
        audio_path="sample_edge.wav",
        language="en",
        on_segment=lambda s: collected.append(s),
    )

    assert len(segments) == 1
    assert segments[0].text == "edge speech recognition works seamlessly"
    assert len(segments[0].words) == 5
    assert lang == "en"
    assert prob == 0.97
    assert len(collected) == 1


def test_factory_resolves_moonshine_variants():
    """Verify EngineRegistry resolves moonshine-tiny and moonshine-base."""
    eng_tiny = get_transcriber("moonshine-tiny", device="cpu")
    assert isinstance(eng_tiny, MoonshineEngine)

    eng_base = get_transcriber("moonshine-base", device="cpu")
    assert isinstance(eng_base, MoonshineEngine)
