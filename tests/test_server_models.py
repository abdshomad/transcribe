"""Tests for server model catalog endpoint and polymorphic pipeline execution."""

from fastapi.testclient import TestClient
from transcribe.server import app


client = TestClient(app)


def test_get_models_catalog_endpoint():
    """Verify GET /api/models returns local and cloud catalog with capabilities."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()

    assert "local" in data
    assert "cloud" in data
    assert len(data["local"]) >= 10
    assert len(data["cloud"]) >= 8

    # Verify model structure and local status
    local_names = [m["name"] for m in data["local"]]
    assert "tiny" in local_names
    assert "sensevoice-small" in local_names
    assert "moonshine-tiny" in local_names
    assert "indonesian-wav2vec2-regional" in local_names

    for m in data["local"]:
        assert m["is_local"] is True
        assert m["implemented"] is True
        assert "family" in m
        assert "params" in m
        assert "display_name" in m
        assert "quantization_options" in m
        assert isinstance(m["quantization_options"], list)
        assert "is_cached" in m
        assert isinstance(m["is_cached"], bool)
        assert "capabilities" in m
        assert isinstance(m["capabilities"], list)

    for m in data["cloud"]:
        assert m["is_local"] is False
        assert m["implemented"] is False
        assert m["is_cached"] is False
