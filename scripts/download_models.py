"""Parallel model downloader using shared server cache and local symlinks."""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import snapshot_download

TARGET_MODELS: Dict[str, Dict[str, str]] = {
    "sensevoice-small": {
        "repo_id": "FunAudioLLM/SenseVoiceSmall",
        "local_dir": "data/models/sensevoice-small",
        "family": "SenseVoice",
    },
    "moonshine-tiny": {
        "repo_id": "UsefulSensors/moonshine-tiny",
        "local_dir": "data/models/moonshine/tiny",
        "family": "Moonshine ONNX",
    },
    "moonshine-base": {
        "repo_id": "UsefulSensors/moonshine-base",
        "local_dir": "data/models/moonshine/base",
        "family": "Moonshine ONNX",
    },
    "meta-mms-1b": {
        "repo_id": "facebook/mms-1b-all",
        "local_dir": "data/models/mms-1b-all",
        "family": "Meta MMS",
    },
    "wav2vec2-regional": {
        "repo_id": "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese",
        "local_dir": "data/models/indonesian-wav2vec2-regional",
        "family": "Indonesian Wav2Vec2",
    },
    "wav2vec2-large-xlsr": {
        "repo_id": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
        "local_dir": "data/models/indonesian-wav2vec2-large-xlsr",
        "family": "Indonesian Wav2Vec2",
    },
}


def download_single_model(model_key: str, info: Dict[str, str]) -> Dict[str, str]:
    """Download model to server HF cache and link into local directory."""
    repo_id = info["repo_id"]
    local_dir = Path(info["local_dir"])
    family = info["family"]

    print(f"🚀 [START] Downloading {family} ({repo_id}) to shared server cache...")
    start_time = time.time()
    try:
        # Download into shared cache (~/.cache/huggingface/hub/)
        cached_snapshot_path = snapshot_download(
            repo_id=repo_id,
            resume_download=True,
        )

        # Create parent directory and symlink
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        if not local_dir.exists():
            try:
                local_dir.symlink_to(cached_snapshot_path, target_is_directory=True)
                print(f"  🔗 Symlinked {local_dir} -> {cached_snapshot_path}")
            except Exception:
                pass

        elapsed = time.time() - start_time
        print(f"✅ [SUCCESS] {family} ({repo_id}) ready in {elapsed:.2f}s (Cache: {cached_snapshot_path})")
        return {"model": model_key, "status": "SUCCESS", "cache_path": cached_snapshot_path, "local_dir": str(local_dir), "elapsed": f"{elapsed:.2f}s"}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ [ERROR] Failed downloading {repo_id}: {e}")
        return {"model": model_key, "status": "FAILED", "error": str(e), "elapsed": f"{elapsed:.2f}s"}


def download_parallel(model_keys: Optional[List[str]] = None, max_workers: int = 3) -> List[Dict[str, str]]:
    """Download multiple models concurrently."""
    selected_keys = model_keys or list(TARGET_MODELS.keys())
    tasks = {k: TARGET_MODELS[k] for k in selected_keys if k in TARGET_MODELS}

    print(f"⚡ Starting parallel download of {len(tasks)} model(s) with {max_workers} worker threads...")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(download_single_model, key, info): key
            for key, info in tasks.items()
        }
        for future in as_completed(future_to_model):
            res = future.result()
            results.append(res)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel ASR Model Downloader to Shared Server Cache")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sensevoice-small", "moonshine-tiny", "meta-mms-1b"],
        help="Model identifiers to download in parallel",
    )
    parser.add_argument("--workers", type=int, default=3, help="Max parallel download workers")
    args = parser.parse_args()

    results = download_parallel(model_keys=args.models, max_workers=args.workers)
    print("\n📊 === Download Summary ===")
    for r in results:
        status_icon = "✅" if r["status"] == "SUCCESS" else "❌"
        print(f"{status_icon} {r['model']}: {r['status']} ({r.get('elapsed', '0s')})")


if __name__ == "__main__":
    main()
