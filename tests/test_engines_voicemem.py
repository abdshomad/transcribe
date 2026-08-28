"""Unit tests for Tsinghua VoiceMem cognitive audio perception engine."""

from unittest.mock import MagicMock, patch
import pytest
from transcribe.engines.base import BaseTranscriber
from transcribe.engines.factory import get_transcriber
from transcribe.engines.voicemem_engine import (
    VoiceMemTranscriber,
    _build_voicemem_words,
    _detect_voicemem_lang,
)


def test_voicemem_words_helper():
    """Verify word timestamp calculations."""
    words = _build_voicemem_words("cognitive voice memory", 3.0)
    assert len(words) == 3
    assert words[0].word == "cognitive"
    assert words[0].start == 0.0
    assert words[0].end == 1.0
    assert words[2].end == 3.0


def test_voicemem_lang_detection():
    """Verify language detection heuristics."""
    assert _detect_voicemem_lang("你好世界", None) == "zh"
    assert _detect_voicemem_lang("ini adalah rekaman", None) == "id"
    assert _detect_voicemem_lang("hello world", None) == "en"
    assert _detect_voicemem_lang("test", "id") == "id"


def test_voicemem_engine_init():
    """Verify transcriber initialization and properties."""
    engine = VoiceMemTranscriber(model_name="voicemem-normal", device="cpu")
    assert isinstance(engine, BaseTranscriber)
    assert engine.model_name == "voicemem-normal"
    assert engine.mode == "normal"


def test_voicemem_factory_resolution():
    """Verify factory instantiation for VoiceMem."""
    engine = get_transcriber("voicemem-normal")
    assert isinstance(engine, VoiceMemTranscriber)


def test_voicemem_mock_transcribe():
    """Verify mocked multimodal ingestion."""
    engine = VoiceMemTranscriber(model_name="voicemem-normal", device="cpu")
    mock_vm = MagicMock()
    mock_vm.ingest.return_value = {
        "text": "Hello Cognitive World",
        "emotion": "happy",
        "scene": "office",
        "speaker": "SPEAKER_01",
    }
    engine._vm = mock_vm

    with patch.object(engine, "_get_audio_duration", return_value=4.0):
        emitted_segs = []
        segs, lang, prob = engine.transcribe(
            "mock.wav",
            on_segment=lambda s: emitted_segs.append(s),
        )
        assert len(segs) == 1
        assert segs[0].text == "Hello Cognitive World"
        assert segs[0].end == 4.0
        assert len(segs[0].words) == 3
        assert len(emitted_segs) == 1
        assert lang == "en"
        assert prob == 0.96
