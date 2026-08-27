from fastapi.testclient import TestClient
from transcribe.server import app
from transcribe.history import save_history, delete_history_item

client = TestClient(app)


def test_api_sources_and_compare():
    src = "comparison_sample.wav"
    job_a = "job_api_tiny_001"
    job_b = "job_api_base_002"

    res_a = {
        "language": "id",
        "duration": 30.0,
        "speakers": ["Speaker 1"],
        "segments": [
            {"id": 0, "speaker": "Speaker 1", "start": 0.0, "end": 3.0, "text": "ini adalah teks pertama"}
        ],
    }
    res_b = {
        "language": "id",
        "duration": 30.0,
        "speakers": ["Speaker 1"],
        "segments": [
            {"id": 0, "speaker": "Speaker 1", "start": 0.0, "end": 3.0, "text": "ini adalah teks pertama yang lengkap"}
        ],
    }

    save_history(job_a, src, "tiny", res_a, processing_time=1.1)
    save_history(job_b, src, "base", res_b, processing_time=3.4)

    # Test /api/sources
    resp = client.get("/api/sources?token=DEMO")
    assert resp.status_code == 200
    sources = resp.json()
    matched = next((s for s in sources if s["source_name"] == src), None)
    assert matched is not None
    assert len(matched["runs"]) >= 2

    # Test /api/compare
    resp = client.get(f"/api/compare?job_a={job_a}&job_b={job_b}&token=DEMO")
    assert resp.status_code == 200
    comp = resp.json()
    assert comp["similarity_score"] > 0
    assert comp["run_a"]["model"] == "tiny"
    assert comp["run_b"]["model"] == "base"
    assert comp["run_a"]["processing_time"] == 1.1
    assert comp["run_b"]["processing_time"] == 3.4
    assert len(comp["run_a"]["diff_words"]) == 4
    assert len(comp["run_b"]["diff_words"]) == 6

    # Clean up
    delete_history_item(job_a)
    delete_history_item(job_b)
