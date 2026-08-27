"""Core universal benchmarking framework across all local ASR model families."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import soundfile as sf
import torch

from .engines.factory import get_transcriber
from .history import save_history
from .metrics import calculate_cer, calculate_rtf, calculate_wer
from .models import MODEL_CATALOG

BENCHMARK_RESULTS_PATH = Path("data/benchmarks/results.json")
BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_benchmark_manifest(manifest_path: str = "data/samples/manifest.json") -> Dict[str, Dict]:
    """Load curated audio samples manifest."""
    p = Path(manifest_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def persist_benchmark_to_history(
    audio_path: Path,
    model_name: str,
    compute_type: str,
    duration: float,
    elapsed: float,
    lang: str,
    lang_prob: float,
    transcript: str,
    segments_raw: list,
) -> None:
    """Save benchmark run to history DB for Web UI visualization."""
    job_id = f"bench_{audio_path.stem}_{model_name.replace('/', '_')}_{compute_type}"
    display_model_name = f"{model_name} ({compute_type})"

    seg_dicts = []
    if segments_raw:
        for s in segments_raw:
            w_list = [w.model_dump() if hasattr(w, "model_dump") else dict(w) for w in getattr(s, "words", []) or []]
            seg_dicts.append({
                "id": getattr(s, "id", 0),
                "start": getattr(s, "start", 0.0),
                "end": getattr(s, "end", duration),
                "text": getattr(s, "text", ""),
                "speaker": "SPEAKER_00",
                "words": w_list,
            })
    else:
        seg_dicts.append({
            "id": 0,
            "start": 0.0,
            "end": duration,
            "text": transcript,
            "speaker": "SPEAKER_00",
            "words": [],
        })

    save_history(
        job_id=job_id,
        source_name=audio_path.name,
        model=display_model_name,
        result_data={
            "language": lang or "unknown",
            "duration": duration,
            "language_probability": lang_prob,
            "speakers": ["SPEAKER_00"],
            "segments": seg_dicts,
        },
        status="completed",
        last_processed_time=duration,
        processing_time=elapsed,
        audio_path=str(audio_path),
    )


def run_model_benchmark(
    audio_path: str,
    model_name: str,
    compute_type: str = "float16",
    device: str = "auto",
    ground_truth: Optional[str] = None,
    language: Optional[str] = None,
    save_db: bool = True,
) -> Dict[str, Any]:
    """Execute single model benchmark recording latency, RTF, WER, CER, VRAM."""
    audio_p = Path(audio_path)
    info = sf.info(str(audio_p))
    audio_dur = info.duration

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    catalog_map = {m.name: m for m in MODEL_CATALOG}
    spec = catalog_map.get(model_name)
    family = spec.family if spec else "ASR"
    params = spec.params if spec else "Unknown"

    start_time = time.perf_counter()
    transcriber = get_transcriber(model_name, device=resolved_device, compute_type=compute_type)
    segments, detected_lang, prob = transcriber.transcribe(str(audio_p), language=language)
    elapsed = time.perf_counter() - start_time

    peak_vram_mb = 0.0
    if resolved_device == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    full_text = " ".join(s.text.strip() for s in segments).strip()
    rtf = calculate_rtf(audio_dur, elapsed)
    wer_val = calculate_wer(ground_truth, full_text) if ground_truth else None
    cer_val = calculate_cer(ground_truth, full_text) if ground_truth else None

    res = {
        "model": model_name,
        "compute_type": compute_type,
        "family": family,
        "params": params,
        "audio_file": audio_p.name,
        "audio_duration_s": round(audio_dur, 2),
        "elapsed_s": round(elapsed, 3),
        "speed_rtf": rtf,
        "peak_vram_mb": peak_vram_mb,
        "detected_lang": detected_lang,
        "lang_prob": round(prob, 3) if prob else 1.0,
        "word_count": len(full_text.split()),
        "wer": wer_val,
        "cer": cer_val,
        "transcript": full_text,
        "status": "SUCCESS",
    }

    if save_db:
        persist_benchmark_to_history(
            audio_p, model_name, compute_type, audio_dur, elapsed, detected_lang, prob, full_text, segments
        )

    return res
