"""Tests for universal benchmark execution framework."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from transcribe.benchmarks import (
    BENCHMARK_RESULTS_PATH,
    load_benchmark_manifest,
    persist_benchmark_to_history,
    run_model_benchmark,
)
from transcribe.models import TranscriptSegment


def test_load_benchmark_manifest():
    """Verify loading manifest data."""
    manifest = load_benchmark_manifest()
    assert isinstance(manifest, dict)
    assert len(manifest) >= 5


@patch("transcribe.benchmarks.get_transcriber")
@patch("soundfile.info")
def test_run_model_benchmark_mocked(mock_sf_info, mock_get_transcriber):
    """Test single model benchmark execution pipeline with mocked transcriber."""
    mock_info = MagicMock()
    mock_info.duration = 10.0
    mock_sf_info.return_value = mock_info

    mock_transcriber = MagicMock()
    mock_seg = TranscriptSegment(
        id=0, start=0.0, end=10.0, text="Hello world testing benchmark pipeline", words=[]
    )
    mock_transcriber.transcribe.return_value = ([mock_seg], "en", 0.99)
    mock_get_transcriber.return_value = mock_transcriber

    res = run_model_benchmark(
        audio_path="data/samples/english_jfk_16k.wav",
        model_name="tiny.en",
        compute_type="float16",
        device="cpu",
        ground_truth="Hello world testing benchmark pipeline",
        save_db=False,
    )

    assert res["status"] == "SUCCESS"
    assert res["model"] == "tiny.en"
    assert res["detected_lang"] == "en"
    assert res["wer"] == 0.0
    assert res["cer"] == 0.0
    assert res["speed_rtf"] > 0
