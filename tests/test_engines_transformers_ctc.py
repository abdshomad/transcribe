"""Tests for TransformersCTCEngine (Wav2Vec2 Regional, Large XLSR & Meta MMS Omnilingual)."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from transcribe.engines.transformers_ctc import (
    TransformersCTCEngine,
    CTC_MODEL_ALIASES,
    normalize_mms_lang,
)
from transcribe.engines.factory import get_transcriber


def test_normalize_mms_lang():
    """Verify ISO-639-1 to ISO-639-3 translation."""
    assert normalize_mms_lang("id") == "ind"
    assert normalize_mms_lang("en") == "eng"
    assert normalize_mms_lang("jv") == "jav"
    assert normalize_mms_lang("su") == "sun"
    assert normalize_mms_lang("fra") == "fra"


def test_ctc_model_aliases():
    """Verify alias mapping for regional Indonesian, MMS and OmniASR models."""
    assert "indonesian-wav2vec2-regional" in CTC_MODEL_ALIASES
    assert "indonesian-wav2vec2-large-xlsr" in CTC_MODEL_ALIASES
    assert "meta-omnilingual-asr" in CTC_MODEL_ALIASES
    assert "meta-mms-300m" in CTC_MODEL_ALIASES
    assert "omniasr-ctc-300m" in CTC_MODEL_ALIASES
    assert CTC_MODEL_ALIASES["omniasr-ctc-300m"] == "bezzam/omniasr-ctc-300m-v2"
    assert CTC_MODEL_ALIASES["meta-omnilingual-asr"] == "facebook/mms-1b-all"


def test_mms_engine_init():
    """Test Meta MMS engine instantiation."""
    engine = TransformersCTCEngine(model_name="meta-omnilingual-asr", target_lang="id", device="cpu")
    assert engine.model_name == "meta-omnilingual-asr"
    assert engine.resolved_model_id in ("facebook/mms-1b-all", "data/models/mms-1b-all")
    assert engine.target_lang == "id"


def test_omniasr_engine_init():
    """Test Meta OmniASR engine instantiation."""
    engine = TransformersCTCEngine(model_name="omniasr-ctc-300m", device="cpu")
    assert engine.model_name == "omniasr-ctc-300m"
    assert engine.resolved_model_id == "bezzam/omniasr-ctc-300m-v2"
    assert get_transcriber("omniasr-ctc-300m") is not None


@patch("soundfile.read")
@patch("transformers.Wav2Vec2Processor.from_pretrained")
@patch("transformers.AutoModelForCTC.from_pretrained")
def test_mms_engine_transcribe_and_adapter_switch(mock_model_cls, mock_proc_cls, mock_sf_read):
    """Test MMS adapter switching during transcription."""
    mock_audio = np.zeros(16000, dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 16000)

    # Mock processor and tokenizer
    mock_processor = MagicMock()
    mock_tokenizer = MagicMock()
    mock_processor.tokenizer = mock_tokenizer
    mock_proc_cls.return_value = mock_processor

    import torch
    mock_inputs = MagicMock()
    mock_inputs.input_values = torch.zeros((1, 16000))
    mock_inputs.attention_mask = None
    mock_processor.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ["bonjour tout le monde"]

    # Mock model
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model_cls.return_value = mock_model
    mock_output = MagicMock()
    mock_output.logits = torch.zeros((1, 50, 32))
    mock_model.return_value = mock_output

    engine = TransformersCTCEngine(model_name="meta-omnilingual-asr", device="cpu")
    segments, lang, prob = engine.transcribe(
        audio_path="french_sample.wav",
        language="fr",
    )

    assert len(segments) == 1
    assert segments[0].text == "bonjour tout le monde"
    assert lang == "fr"
    assert prob == 0.95

    # Check adapter switching was invoked with 3-letter MMS code 'fra'
    mock_tokenizer.set_target_lang.assert_called_with("fra")
    mock_model.load_adapter.assert_called_with("fra")


def test_factory_resolves_meta_mms():
    """Verify EngineRegistry routes meta-omnilingual-asr to TransformersCTCEngine."""
    engine = get_transcriber("meta-omnilingual-asr", device="cpu")
    assert isinstance(engine, TransformersCTCEngine)
    assert engine.resolved_model_id in ("facebook/mms-1b-all", "data/models/mms-1b-all")
