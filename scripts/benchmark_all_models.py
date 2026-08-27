#!/usr/bin/env python3
"""
Comprehensive Multi-Model, Multi-Size & Multi-Variant ASR Benchmark Suite
Defaults to exhaustive evaluation:
- Default Models: ALL 19 Architectures (Standard, English .en, Distil, Cahya Indonesian)
- Default Sizes: ALL 5 Tiers (Tiny, Base, Small, Medium, Large/Turbo)
- Default Quantization Variants: ALL 3 Modes (float16, int8_float16, int8)
- Default Datasets: ALL Samples (proklamasi.wav & jfk.wav)
- Incremental Caching & SQLite Database Persistence enabled by default
"""

import time
import os
import sys
import json
import argparse
import torch
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root and src to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from transcribe.models import MODEL_CATALOG, ASRModelInfo
from transcribe.transcriber import FasterWhisperTranscriber
from transcribe.history import save_history
from transcribe.metrics import normalize_text, calculate_wer, calculate_cer, calculate_rtf

SIZE_TIERS = {
    "tiny": ["tiny", "tiny.en", "cahya-whisper-tiny-id"],
    "base": ["base", "base.en"],
    "small": ["small", "small.en", "distil-small.en", "cahya-whisper-small-id"],
    "medium": ["medium", "medium.en", "distil-medium.en", "cahya-whisper-medium-id"],
    "large": ["large-v1", "large-v2", "large-v3", "turbo", "distil-large-v2", "distil-large-v3"],
}

DEFAULT_MODELS = [
    # Tiny Tier (39M)
    "tiny", "tiny.en", "cahya-whisper-tiny-id",
    # Base Tier (74M)
    "base", "base.en",
    # Small Tier (166M - 244M)
    "small", "small.en", "distil-small.en", "cahya-whisper-small-id",
    # Medium Tier (394M - 769M)
    "medium", "medium.en", "distil-medium.en", "cahya-whisper-medium-id",
    # Large & Turbo Tier (756M - 1550M)
    "large-v1", "large-v2", "large-v3", "turbo", "distil-large-v2", "distil-large-v3"
]

DEFAULT_COMPUTE_TYPES = ["float16", "int8_float16", "int8"]

KNOWN_GROUND_TRUTHS = {
    "proklamasi.wav": (
        "Kami bangsa Indonesia dengan ini menjatakan kemerdekaan Indonesia. "
        "Hal-hal jang mengenai pemindahan kekoeasaan d.l.l., diselenggarakan "
        "dengan tjara saksama dan dalam tempo jang sesingkat-singkatnja. "
        "Djakarta, hari 17 boelan 8 tahoen 05. Atas nama bangsa Indonesia, Soekarno Hatta."
    ),
    "jfk.wav": (
        "And so, my fellow Americans: ask not what your country can do for you—"
        "ask what you can do for your country."
    )
}

DEFAULT_SAMPLES = [
    WORKSPACE_ROOT / "data" / "sample" / "proklamasi.wav",
    WORKSPACE_ROOT / "data" / "sample" / "jfk.wav"
]


def get_size_tier_label(model_name: str) -> str:
    """Return human-readable size tier for model."""
    for tier, models in SIZE_TIERS.items():
        if model_name in models:
            return tier.capitalize()
    return "Custom"


def load_cached_results(json_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing successful benchmark results keyed by (name, compute_type)."""
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cached_list = data.get("results", [])
            return {
                f"{r['name']}__{r.get('compute_type', 'float16')}": r
                for r in cached_list
                if r.get("status") == "SUCCESS" and r.get("elapsed_s", 0) > 0
            }
    except Exception:
        return {}


def persist_to_db(audio_path: Path, model_name: str, compute_type: str, duration: float, elapsed: float, lang: str, lang_prob: float, transcript: str, segments_raw: list):
    """Save benchmark run to SQLite data/history.db so it appears on Web UI screen."""
    sample_slug = audio_path.stem
    clean_model_key = model_name.replace("/", "_")
    job_id = f"bench_{sample_slug}_{clean_model_key}_{compute_type}"
    display_model_name = f"{model_name} ({compute_type})"
    
    seg_dicts = []
    if segments_raw:
        for s in segments_raw:
            if hasattr(s, "words"):
                w_list = [w.model_dump() if hasattr(w, "model_dump") else dict(w) for w in (s.words or [])]
            elif isinstance(s, dict):
                w_list = s.get("words", [])
            else:
                w_list = []
            
            s_text = s.text if hasattr(s, "text") else (s.get("text", "") if isinstance(s, dict) else "")
            s_start = s.start if hasattr(s, "start") else (s.get("start", 0.0) if isinstance(s, dict) else 0.0)
            s_end = s.end if hasattr(s, "end") else (s.get("end", duration) if isinstance(s, dict) else duration)
            s_id = s.id if hasattr(s, "id") else (s.get("id", 0) if isinstance(s, dict) else 0)
            
            seg_dicts.append({
                "id": s_id,
                "start": s_start,
                "end": s_end,
                "text": s_text,
                "speaker": "SPEAKER_00",
                "words": w_list
            })
    else:
        seg_dicts.append({
            "id": 0,
            "start": 0.0,
            "end": duration,
            "text": transcript,
            "speaker": "SPEAKER_00",
            "words": []
        })

    result_data = {
        "language": lang or "unknown",
        "duration": duration,
        "language_probability": lang_prob,
        "speakers": ["SPEAKER_00"],
        "segments": seg_dicts
    }
    
    save_history(
        job_id=job_id,
        source_name=audio_path.name,
        model=display_model_name,
        result_data=result_data,
        status="completed",
        last_processed_time=duration,
        processing_time=elapsed,
        audio_path=str(audio_path),
    )


def benchmark_sample(
    audio_path: Path,
    models_to_test: List[str],
    compute_types: List[str],
    device: str = "auto",
    force: bool = False,
    save_db: bool = True
) -> Dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    audio_info = sf.info(str(audio_path))
    audio_duration = audio_info.duration
    
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    gpu_name = torch.cuda.get_device_name(0) if resolved_device == "cuda" else "CPU"
    
    ground_truth = KNOWN_GROUND_TRUTHS.get(audio_path.name)
    
    sample_slug = audio_path.stem
    report_file = WORKSPACE_ROOT / "docs" / "features" / "core" / f"all-models-benchmark-{sample_slug}.md"
    json_file = report_file.with_suffix(".json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    cached_map = {} if force else load_cached_results(json_file)
    
    total_runs = len(models_to_test) * len(compute_types)
    print("\n" + "="*85)
    print(f"🎙️ BENCHMARKING AUDIO: {audio_path.name} ({audio_duration:.2f}s, {audio_info.samplerate}Hz)")
    print(f"🖥️ Hardware: {gpu_name} | Models: {len(models_to_test)} | Quantizations: {compute_types} (Total Runs: {total_runs})")
    if cached_map:
        print(f"⚡ Cache: Found {len(cached_map)} cached model/quantization runs (use --force to re-run all)")
    if save_db:
        print(f"💾 Database: Results will be saved to SQLite `data/history.db` for Web UI comparison")
    if ground_truth:
        print(f"🎯 Ground Truth: \"{ground_truth[:65]}...\"")
    print("="*85)
    
    catalog_map = {m.name: m for m in MODEL_CATALOG}
    results = []
    run_idx = 0
    
    for ctype in compute_types:
        for model_name in models_to_test:
            run_idx += 1
            spec = catalog_map.get(model_name)
            family = spec.family if spec else "Whisper"
            params = spec.params if spec else "Unknown"
            vram_est = spec.vram if spec else "Unknown"
            size_tier = get_size_tier_label(model_name)
            
            cache_key = f"{model_name}__{ctype}"
            
            # Check cache
            if cache_key in cached_map and not force:
                cached_entry = cached_map[cache_key]
                cached_entry["size_tier"] = size_tier
                results.append(cached_entry)
                if save_db:
                    persist_to_db(
                        audio_path=audio_path,
                        model_name=model_name,
                        compute_type=ctype,
                        duration=audio_duration,
                        elapsed=cached_entry.get("elapsed_s", 1.0),
                        lang=cached_entry.get("detected_lang", "unknown"),
                        lang_prob=cached_entry.get("lang_prob", 1.0),
                        transcript=cached_entry.get("transcript", ""),
                        segments_raw=cached_entry.get("segments", [])
                    )
                wer_str = f", WER: {cached_entry.get('wer'):>5.1f}%" if cached_entry.get('wer') is not None else ""
                print(f"[{run_idx:02d}/{total_runs}] Skipped: {model_name:<28} [{ctype:<12}] (Cached ⚡ {cached_entry['elapsed_s']:.2f}s, {cached_entry['speed_rtf']:>5.1f}x RTF{wer_str})")
                continue
                
            print(f"[{run_idx:02d}/{total_runs}] Running: {model_name:<28} [{ctype:<12}] ({size_tier:<6} | {params:>5})...", end="", flush=True)
            
            try:
                if resolved_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    
                transcriber = FasterWhisperTranscriber(model_size=model_name, device=resolved_device, compute_type=ctype)
                
                start_time = time.perf_counter()
                segments, lang, lang_prob = transcriber.transcribe(
                    str(audio_path),
                    language=None,
                    beam_size=5,
                    vad_filter=True
                )
                elapsed = time.perf_counter() - start_time
                
                rtf = calculate_rtf(audio_duration, elapsed)
                
                peak_vram_mb = 0
                if resolved_device == "cuda":
                    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                    
                full_text = " ".join(s.text.strip() for s in segments).strip()
                word_count = len(full_text.split())
                char_count = len(full_text)
                lang_prob_val = round(lang_prob, 3) if lang_prob else 1.0
                
                wer_val = calculate_wer(ground_truth, full_text) if ground_truth else None
                cer_val = calculate_cer(ground_truth, full_text) if ground_truth else None
                
                seg_dicts = [
                    {
                        "id": s.id,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "words": [w.model_dump() if hasattr(w, "model_dump") else dict(w) for w in (s.words or [])]
                    }
                    for s in segments
                ]
                
                res_entry = {
                    "name": model_name,
                    "compute_type": ctype,
                    "size_tier": size_tier,
                    "family": family,
                    "parameters": params,
                    "vram_est": vram_est,
                    "elapsed_s": round(elapsed, 3),
                    "audio_duration_s": round(audio_duration, 2),
                    "speed_rtf": rtf,
                    "peak_vram_mb": peak_vram_mb,
                    "detected_lang": lang or "unknown",
                    "lang_prob": lang_prob_val,
                    "word_count": word_count,
                    "char_count": char_count,
                    "wer": wer_val,
                    "cer": cer_val,
                    "transcript": full_text,
                    "segments": seg_dicts,
                    "status": "SUCCESS"
                }
                results.append(res_entry)
                
                if save_db:
                    persist_to_db(
                        audio_path=audio_path,
                        model_name=model_name,
                        compute_type=ctype,
                        duration=audio_duration,
                        elapsed=elapsed,
                        lang=lang or "unknown",
                        lang_prob=lang_prob_val,
                        transcript=full_text,
                        segments_raw=segments
                    )
                
                wer_str = f", WER: {wer_val:>5.1f}%" if wer_val is not None else ""
                print(f" ✅ {elapsed:.2f}s ({rtf:>5.1f}x RTF{wer_str}, {word_count:>2}w, lang={lang} [{lang_prob_val:.2f}])")
                
                del transcriber
                if resolved_device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f" ❌ FAILED: {e}")
                results.append({
                    "name": model_name,
                    "compute_type": ctype,
                    "size_tier": size_tier,
                    "family": family,
                    "parameters": params,
                    "vram_est": vram_est,
                    "elapsed_s": 0,
                    "audio_duration_s": round(audio_duration, 2),
                    "speed_rtf": 0,
                    "peak_vram_mb": 0,
                    "detected_lang": "error",
                    "lang_prob": 0,
                    "word_count": 0,
                    "char_count": 0,
                    "wer": 100.0,
                    "cer": 100.0,
                    "transcript": f"Error: {str(e)}",
                    "segments": [],
                    "status": f"FAILED: {e}"
                })
                
    # Generate Markdown Report
    md_lines = [
        f"# Comprehensive Benchmark Report: `{audio_path.name}`",
        "",
        f"> **Hardware**: {gpu_name}  ",
        f"> **Audio Sample**: `{audio_path.name}` (Duration: {audio_duration:.2f}s)  ",
        f"> **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Engine**: Faster-Whisper (CTranslate2) across {len(compute_types)} Quantization Modes",
        ""
    ]
    if ground_truth:
        md_lines.extend([
            f"> **Ground Truth Reference**: \"{ground_truth}\"",
            ""
        ])
    md_lines.extend([
        "---",
        "",
        "## 1. Summary Comparison Matrix",
        ""
    ])
    
    if ground_truth:
        md_lines.append("| Rank | Model | Size Tier | Quantization | Family | Params | WER (%) | CER (%) | Time (s) | Speed (x RTF) | Lang (Prob) | Words |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        sorted_results = sorted([r for r in results if r["status"] == "SUCCESS"], key=lambda x: (x["wer"] if x["wer"] is not None else 999, -x["speed_rtf"]))
        for rank, r in enumerate(sorted_results, start=1):
            tier_str = r.get("size_tier", get_size_tier_label(r['name']))
            md_lines.append(
                f"| {rank} | **`{r['name']}`** | **{tier_str}** | `{r.get('compute_type', 'float16')}` | {r['family']} | {r['parameters']} | "
                f"**{r['wer']}%** | {r['cer']}% | {r['elapsed_s']}s | **{r['speed_rtf']}x** | `{r['detected_lang']}` ({r['lang_prob']}) | {r['word_count']} |"
            )
    else:
        md_lines.append("| Rank | Model | Size Tier | Quantization | Family | Params | Time (s) | Speed (x RTF) | Lang (Prob) | Words |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        sorted_results = sorted([r for r in results if r["status"] == "SUCCESS"], key=lambda x: -x["speed_rtf"])
        for rank, r in enumerate(sorted_results, start=1):
            tier_str = r.get("size_tier", get_size_tier_label(r['name']))
            md_lines.append(
                f"| {rank} | **`{r['name']}`** | **{tier_str}** | `{r.get('compute_type', 'float16')}` | {r['family']} | {r['parameters']} | "
                f"{r['elapsed_s']}s | **{r['speed_rtf']}x** | `{r['detected_lang']}` ({r['lang_prob']}) | {r['word_count']} |"
            )
            
    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Performance Breakdown by Size Tier",
        ""
    ])
    
    for tier in ["Tiny", "Base", "Small", "Medium", "Large"]:
        tier_results = [r for r in sorted_results if r.get("size_tier") == tier]
        if tier_results:
            md_lines.extend([
                f"### Tier: {tier}",
                "| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |"
            ])
            for r in tier_results:
                wer_str = f"**{r['wer']}%**" if r.get('wer') is not None else "--"
                md_lines.append(f"| `{r['name']}` | `{r.get('compute_type', 'float16')}` | {wer_str} | **{r['speed_rtf']}x** | {r['elapsed_s']}s | `{r['detected_lang']}` |")
            md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 3. Full Transcripts Generated per Model Variant",
        ""
    ])
    
    for r in results:
        wer_tag = f" | **WER**: {r['wer']}%" if r.get("wer") is not None else ""
        tier_tag = r.get("size_tier", get_size_tier_label(r['name']))
        md_lines.extend([
            f"### `{r['name']}` [{r.get('compute_type', 'float16')}] ({tier_tag} — {r['parameters']})",
            f"- **Speed**: {r['speed_rtf']}x RTF ({r['elapsed_s']}s){wer_tag} | **Language**: `{r['detected_lang']}` (prob: {r['lang_prob']}) | **Words**: {r['word_count']}",
            "```text",
            r['transcript'],
            "```",
            ""
        ])
        
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "audio_file": str(audio_path),
            "duration": audio_duration,
            "device": gpu_name,
            "compute_types": compute_types,
            "ground_truth": ground_truth,
            "results": results
        }, f, indent=2)
        
    print(f"\n📊 Report updated: {report_file}")
    return {
        "sample": audio_path.name,
        "report": str(report_file),
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Model, Multi-Size & Multi-Variant ASR Benchmark Suite.")
    parser.add_argument("--audio", type=str, default=None, help="Path to single audio file (defaults to all sample datasets)")
    parser.add_argument("--sizes", nargs="+", default=["all"], choices=["tiny", "base", "small", "medium", "large", "all"], help="Filter benchmark by size tiers (default: all)")
    parser.add_argument("--models", nargs="+", default=None, help="Explicit list of model names to benchmark (default: all)")
    parser.add_argument("--compute-types", nargs="+", default=DEFAULT_COMPUTE_TYPES, choices=["float16", "int8_float16", "int8", "float32", "all"], help="Compute quantization precisions (default: float16 int8_float16 int8)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--force", action="store_true", help="Force re-run all models, bypassing cache")
    parser.add_argument("--no-db", action="store_true", help="Disable saving results to SQLite data/history.db")
    
    args = parser.parse_args()
    
    # Resolve target models by size tiers
    if args.models:
        target_models = args.models
    elif "all" in args.sizes:
        target_models = DEFAULT_MODELS
    else:
        target_models = []
        for s in args.sizes:
            target_models.extend(SIZE_TIERS.get(s, []))
            
    # Resolve compute types
    if "all" in args.compute_types:
        compute_types = ["float16", "int8_float16", "int8"]
    else:
        compute_types = args.compute_types
        
    # Resolve samples
    if args.audio:
        samples = [Path(args.audio)]
    else:
        samples = DEFAULT_SAMPLES
        
    for s in samples:
        benchmark_sample(
            s,
            target_models,
            compute_types=compute_types,
            device=args.device,
            force=args.force,
            save_db=not args.no_db
        )


if __name__ == "__main__":
    main()
