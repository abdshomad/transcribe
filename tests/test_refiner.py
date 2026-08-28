"""Unit tests for AI Transcript Refiner layer and FreeToken integration."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from transcribe.refiner import (
    DEFAULT_REFINER_PROMPT,
    format_transcript_for_refinement,
    refine_transcript_sync,
    refine_transcript_stream,
)
from transcribe.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_format_transcript_for_refinement():
    """Verify segment consolidation for refinement."""
    segs = [
        {"speaker": "Alice", "text": "um so like good morning."},
        {"speaker": "Alice", "text": "we should start the sync."},
        {"speaker": "Bob", "text": "yes i am ready."},
    ]
    formatted = format_transcript_for_refinement(segs)
    assert "Alice: um so like good morning. we should start the sync." in formatted
    assert "Bob: yes i am ready." in formatted


def test_refine_transcript_sync_empty():
    """Verify empty input handling."""
    res = refine_transcript_sync([])
    assert "Empty transcript" in res


def test_refine_transcript_sync_mocked():
    """Verify mocked sync refinement call."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Alice: Good morning. We should start the sync.\n\nBob: Yes, I am ready."}}]
    }
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_resp):
        segs = [{"speaker": "Alice", "text": "um so like good morning."}]
        res = refine_transcript_sync(segs)
        assert "Good morning." in res


@pytest.mark.anyio
async def test_refine_transcript_stream_mocked():
    """Verify mocked async streaming refinement generator."""
    lines = [
        b"data: {\"choices\": [{\"delta\": {\"content\": \"Alice: Good morning. \"}}]}\n",
        b"data: {\"choices\": [{\"delta\": {\"content\": \"We should begin.\"}}]}\n",
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
        segs = [{"speaker": "Alice", "text": "um so like good morning."}]
        chunks = []
        async for chunk in refine_transcript_stream(segs):
            chunks.append(chunk)

        assert "".join(chunks) == "Alice: Good morning. We should begin."


def test_api_refine_endpoint(client):
    """Verify POST /api/refine with test token."""
    mock_polished = "Alice: Hello everyone."
    with patch("transcribe.server.refine_transcript_sync", return_value=mock_polished):
        resp = client.post(
            "/api/refine?token=DEMO",
            json={
                "segments": [{"speaker": "Alice", "text": "um hello everyone"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["refined_text"] == mock_polished
