"""Live inference tests executing real downloaded model weights on server GPU/CPU."""

import os
from pathlib import Path
import pytest
from transcribe.engines.factory import get_transcriber

SAMPLE_INDONESIAN = "data/samples/indonesian_proklamasi_16k.wav"
SAMPLE_ENGLISH = "data/samples/english_jfk_16k.wav"
SAMPLE_CHINESE = "data/samples/chinese_speech_16k.wav"


def sample_available(path: str) -> bool:
    return Path(path).exists()


@pytest.mark.skipif(not sample_available(SAMPLE_INDONESIAN), reason="Indonesian sample not curated")
def test_live_faster_whisper_tiny():
    """Live inference test for faster-whisper tiny on local audio."""
    engine = get_transcriber("tiny", device="auto")
    segments, lang, prob = engine.transcribe(SAMPLE_INDONESIAN)

    assert len(segments) > 0
    full_text = " ".join(s.text.strip() for s in segments)
    assert len(full_text) > 10
    assert lang in ["id", "jw", "ms", "en"]
    assert prob > 0.4


@pytest.mark.skipif(not sample_available(SAMPLE_ENGLISH), reason="English sample not curated")
def test_live_faster_whisper_english():
    """Live inference test for faster-whisper on English speech."""
    engine = get_transcriber("tiny.en", device="auto")
    segments, lang, prob = engine.transcribe(SAMPLE_ENGLISH)

    assert len(segments) > 0
    full_text = " ".join(s.text.strip() for s in segments)
    assert any(k in full_text.lower() for k in ["country", "americans", "ask", "what"])
    assert lang == "en"


@pytest.mark.skipif(not sample_available(SAMPLE_INDONESIAN), reason="Indonesian sample not curated")
def test_live_indonesian_wav2vec2_regional():
    """Live inference test for Indonesian Wav2Vec2 regional model."""
    engine = get_transcriber("indonesian-wav2vec2-regional", device="auto")
    segments, lang, prob = engine.transcribe(SAMPLE_INDONESIAN, language="id")

    assert len(segments) > 0
    full_text = " ".join(s.text.strip() for s in segments)
    assert len(full_text) > 10
    assert lang == "id"


@pytest.mark.skipif(not sample_available(SAMPLE_CHINESE), reason="Chinese sample not curated")
def test_live_sensevoice_tag_extraction():
    """Live inference test for SenseVoice-Small with SER and AED tag parsing."""
    engine = get_transcriber("sensevoice-small", device="auto")
    segments, lang, prob = engine.transcribe(SAMPLE_CHINESE, language="zh")

    assert len(segments) > 0
    assert lang == "zh"
    assert segments[0].text is not None


@pytest.mark.skipif(not sample_available(SAMPLE_ENGLISH), reason="English sample not curated")
def test_live_moonshine_base_inference():
    """Live inference test for UsefulSensors Moonshine edge model."""
    engine = get_transcriber("moonshine-base", device="auto")
    segments, lang, prob = engine.transcribe(SAMPLE_ENGLISH)

    assert len(segments) > 0
    assert segments[0].text is not None
    assert len(segments[0].words) > 0
