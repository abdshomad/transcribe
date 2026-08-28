"""Unit tests for Minutes of Meeting (MOM) generation and FreeToken integration."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from transcribe.mom import (
    DEFAULT_MOM_PROMPT,
    format_transcript_for_mom,
    _get_llm_config,
    generate_mom_sync,
    generate_mom_stream,
)
from transcribe.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_format_transcript_for_mom():
    """Verify speaker and turn aggregation."""
    segs = [
        {"speaker": "Alice", "text": "Good morning everyone."},
        {"speaker": "Alice", "text": "Let's review the Q3 roadmap."},
        {"speaker": "Bob", "text": "I have updated the backend metrics."},
        {"speaker": None, "text": "Any other questions?"},
    ]
    formatted = format_transcript_for_mom(segs)
    assert "Alice: Good morning everyone. Let's review the Q3 roadmap." in formatted
    assert "Bob: I have updated the backend metrics." in formatted
    assert "Speaker: Any other questions?" in formatted


def test_get_llm_config_defaults():
    """Verify config defaults."""
    cfg = _get_llm_config()
    assert "base_url" in cfg
    assert "api_key" in cfg
    assert "model" in cfg


def test_generate_mom_sync_empty():
    """Verify empty input handling."""
    res = generate_mom_sync([])
    assert "Empty transcript" in res


def test_generate_mom_sync_mocked():
    """Verify mocked sync LLM call."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "## 1. Meeting Overview\n* **Topic:** Q3 Review"}}]
    }
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_resp):
        segs = [{"speaker": "Alice", "text": "Meeting content."}]
        res = generate_mom_sync(segs)
        assert "## 1. Meeting Overview" in res


@pytest.mark.anyio
async def test_generate_mom_stream_mocked():
    """Verify mocked async streaming generator."""
    lines = [
        b"data: {\"choices\": [{\"delta\": {\"content\": \"## Executive Summary\\n\"}}]}\n",
        b"data: {\"choices\": [{\"delta\": {\"content\": \"Meeting finalized.\"}}]}\n",
        b"data: [DONE]\n",
    ]

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None

    async def _mock_lines():
        for line in lines:
            yield line.decode("utf-8")

    mock_resp.aiter_lines = _mock_lines

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def stream(self, *args, **kwargs):
            class StreamCtx:
                async def __aenter__(self):
                    return mock_resp

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            return StreamCtx()

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        segs = [{"speaker": "Alice", "text": "Meeting dialogue"}]
        chunks = []
        async for chunk in generate_mom_stream(segs):
            chunks.append(chunk)

        assert "".join(chunks) == "## Executive Summary\nMeeting finalized."


def test_api_mom_endpoint(client):
    """Verify POST /api/mom with test token."""
    mock_mom = "## 1. Meeting Overview\nAll tasks assigned."
    with patch("transcribe.server.generate_mom_sync", return_value=mock_mom):
        resp = client.post(
            "/api/mom?token=DEMO",
            json={
                "segments": [{"speaker": "Alice", "text": "Hello world"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["mom_markdown"] == mock_mom
