"""Tests for SenseVoiceEngine (Alibaba FunAudioLLM SenseVoiceSmall)."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from transcribe.engines.sensevoice import SenseVoiceEngine
from transcribe.engines.factory import get_transcriber


def test_sensevoice_engine_init():
    """Test engine instantiation and model default path."""
    engine = SenseVoiceEngine(model_name="sensevoice-small", device="cpu")
    assert engine.model_name == "sensevoice-small"
    assert "sensevoice" in engine.resolved_model_path.lower()
    assert engine.device == "cpu"


def test_sensevoice_tag_extraction():
    """Test extracting language, emotion, and audio event detection tags."""
    engine = SenseVoiceEngine()
    raw = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好，世界！"
    clean, tags = engine.clean_text_and_extract_tags(raw)
    assert clean == "你好，世界！"
    assert tags.get("language") == "zh"
    assert tags.get("emotion") == "NEUTRAL"

    raw_emotion_event = "<|en|><|HAPPY|><|LAUGHTER|><|APPLAUSE|>This is wonderful!"
    clean_ev, tags_ev = engine.clean_text_and_extract_tags(raw_emotion_event)
    assert clean_ev == "This is wonderful!"
    assert tags_ev.get("language") == "en"
    assert tags_ev.get("emotion") == "HAPPY"
    assert "LAUGHTER" in tags_ev.get("events", "")
    assert "APPLAUSE" in tags_ev.get("events", "")

    raw_sad_music = "<|en|><|SAD|><|MUSIC|><|CRY|>Melancholy tune"
    clean_sad, tags_sad = engine.clean_text_and_extract_tags(raw_sad_music)
    assert clean_sad == "Melancholy tune"
    assert tags_sad.get("emotion") == "SAD"
    assert "MUSIC" in tags_sad.get("events", "")
    assert "CRY" in tags_sad.get("events", "")


@patch("soundfile.read")
def test_sensevoice_transcribe_mock_model(mock_sf_read):
    """Test transcribe flow with mock model generator and emotion/event population."""
    mock_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds
    mock_sf_read.return_value = (mock_audio, 16000)

    engine = SenseVoiceEngine(device="cpu")
    mock_model = MagicMock()
    mock_model.generate.return_value = [
        {"text": "<|en|><|HAPPY|><|LAUGHTER|><|Speech|>Hello world welcome to SenseVoice"}
    ]
    engine._model = mock_model

    collected = []
    segments, lang, prob = engine.transcribe(
        audio_path="test_speech.wav",
        language="en",
        on_segment=lambda s: collected.append(s),
    )

    assert len(segments) == 1
    assert segments[0].text == "Hello world welcome to SenseVoice"
    assert segments[0].end == 2.0
    assert len(segments[0].words) == 5
    assert segments[0].emotion == "HAPPY"
    assert "LAUGHTER" in segments[0].events
    assert lang == "en"
    assert prob == 0.98
    assert len(collected) == 1


def test_factory_resolves_sensevoice():
    """Verify EngineRegistry resolves sensevoice-small to SenseVoiceEngine."""
    engine = get_transcriber("sensevoice-small", device="cpu")
    assert isinstance(engine, SenseVoiceEngine)
