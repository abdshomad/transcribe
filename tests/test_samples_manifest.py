"""Tests for curated multilingual test audio samples and manifest."""

import json
from pathlib import Path
import soundfile as sf


def test_samples_manifest_integrity():
    manifest_path = Path("data/samples/manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest) >= 5
    for sample_id, meta in manifest.items():
        sample_path = Path(meta["path"])
        assert sample_path.exists(), f"Sample audio missing: {sample_path}"
        info = sf.info(str(sample_path))
        assert info.samplerate == 16000, f"Sample {sample_id} not 16kHz: {info.samplerate}"
        assert info.channels == 1, f"Sample {sample_id} not mono: {info.channels}"
        assert meta["duration"] > 0
        assert meta["language"] in ["id", "en", "zh", "ja", "yue", "jv", "su"]
