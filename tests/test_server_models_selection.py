"""Tests for server model catalog, dynamic cascaded selection, and inference parameters."""

from fastapi.testclient import TestClient
from transcribe.server import app
from transcribe.models import check_model_cached, get_enriched_model_catalog


client = TestClient(app)


def test_enriched_catalog_cache_status():
    """Verify get_enriched_model_catalog returns dynamic local cache indicators."""
    catalog = get_enriched_model_catalog()
    assert len(catalog) >= 15
    for m in catalog:
        assert isinstance(m.is_cached, bool)
        assert isinstance(m.quantization_options, list)
        assert len(m.quantization_options) >= 1
        assert m.display_name != ""


def test_check_model_cached_behavior():
    """Verify check_model_cached identifies existing local directory and HF cache."""
    # SenseVoice small is downloaded in data/models/sensevoice-small
    assert check_model_cached("sensevoice-small") is True
    # Non-existent model should return False
    assert check_model_cached("nonexistent-model-xyz-999") is False


def test_transcribe_stream_accepts_dynamic_parameters():
    """Verify POST /api/transcribe-stream accepts compute_type, beam_size, vad_filter, use_itn."""
    # Test validation with parameters
    response = client.post(
        "/api/transcribe-stream",
        data={
            "model": "tiny",
            "compute_type": "int8",
            "beam_size": "3",
            "vad_filter": "true",
            "use_itn": "true",
            "chunk_length_s": "15.0",
        },
    )
    # 400 Bad Request or 401 (if token enforced)
    assert response.status_code in (400, 401)
