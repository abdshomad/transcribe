#!/usr/bin/env python3
"""Universal ASR Benchmark CLI for all polymorphic models, sizes & precisions."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from transcribe.benchmarks import BENCHMARK_RESULTS_PATH, load_benchmark_manifest, run_model_benchmark

DEFAULT_BENCHMARK_MODELS = [
    # Faster-Whisper
    "tiny",
    "base",
    "small",
    "medium",
    "turbo",
    "large-v3",
    "distil-small.en",
    # Indonesian Regional & XLSR
    "indonesian-wav2vec2-regional",
    "indonesian-wav2vec2-large-xlsr",
    # Alibaba SenseVoice
    "sensevoice-small",
    # UsefulSensors Moonshine
    "moonshine-tiny",
    "moonshine-base",
]

DEFAULT_COMPUTE_TYPES = ["float16", "int8", "float32"]


def save_master_results(new_results: List[Dict[str, Any]]) -> None:
    """Append or update master results in data/benchmarks/results.json."""
    existing_data: Dict[str, Any] = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "benchmarks": {}}
    if BENCHMARK_RESULTS_PATH.exists():
        try:
            existing_data = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    bench_dict = existing_data.get("benchmarks", {})
    for r in new_results:
        key = f"{r['audio_file']}__{r['model']}__{r.get('compute_type', 'default')}"
        bench_dict[key] = r

    existing_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    existing_data["benchmarks"] = bench_dict
    BENCHMARK_RESULTS_PATH.write_text(json.dumps(existing_data, indent=2), encoding="utf-8")


def generate_markdown_report(results: List[Dict[str, Any]], audio_name: str, out_path: Path) -> None:
    """Generate structured markdown summary matrix."""
    lines = [
        f"# 📊 Universal ASR Benchmark Report: `{audio_name}`",
        "",
        f"> **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Total Models Evaluated**: {len(results)}",
        "",
        "---",
        "",
        "| Rank | Model | Family | Precision | Time (s) | Speed (x RTF) | WER (%) | CER (%) | Lang | Peak VRAM |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    sorted_res = sorted([r for r in results if r["status"] == "SUCCESS"], key=lambda x: (x.get("wer") or 999, -x["speed_rtf"]))
    for rank, r in enumerate(sorted_res, start=1):
        wer_s = f"**{r['wer']}%**" if r.get("wer") is not None else "--"
        cer_s = f"{r['cer']}%" if r.get("cer") is not None else "--"
        vram_s = f"{r['peak_vram_mb']} MB" if r.get("peak_vram_mb") else "--"
        lines.append(
            f"| {rank} | **`{r['model']}`** | {r['family']} | `{r['compute_type']}` | {r['elapsed_s']}s | "
            f"**{r['speed_rtf']}x** | {wer_s} | {cer_s} | `{r['detected_lang']}` | {vram_s} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 Report written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Polymorphic ASR Benchmark Suite")
    parser.add_argument("--models", nargs="+", default=DEFAULT_BENCHMARK_MODELS, help="Models to benchmark")
    parser.add_argument("--compute-types", nargs="+", default=["float16", "int8"], help="Quantizations")
    parser.add_argument("--audio", type=str, default=None, help="Explicit audio file")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--no-db", action="store_true", help="Skip database persistence")
    args = parser.parse_args()

    manifest = load_benchmark_manifest()
    sample_list = []
    if args.audio:
        sample_list.append({"path": args.audio, "name": Path(args.audio).name, "ground_truth": None, "lang": None})
    elif manifest:
        for item in manifest.values():
            sample_list.append({
                "path": item["path"],
                "name": Path(item["path"]).name,
                "ground_truth": item.get("ground_truth"),
                "lang": item.get("language"),
            })
    else:
        sample_list.append({
            "path": "data/sample/proklamasi.wav",
            "name": "proklamasi.wav",
            "ground_truth": "Kami bangsa Indonesia dengan ini menjatakan kemerdekaan Indonesia.",
            "lang": "id",
        })

    all_results = []
    for s in sample_list:
        sample_path = s["path"]
        if not Path(sample_path).exists():
            continue
        print(f"\n🎧 Benchmarking Sample: {s['name']}...")
        sample_results = []
        for model in args.models:
            for ctype in args.compute_types:
                print(f"  ▶ [{model}] [{ctype}]... ", end="", flush=True)
                try:
                    res = run_model_benchmark(
                        audio_path=sample_path,
                        model_name=model,
                        compute_type=ctype,
                        device=args.device,
                        ground_truth=s.get("ground_truth"),
                        language=s.get("lang"),
                        save_db=not args.no_db,
                    )
                    sample_results.append(res)
                    all_results.append(res)
                    wer_str = f" | WER: {res['wer']}%" if res.get("wer") is not None else ""
                    print(f"✅ {res['elapsed_s']}s ({res['speed_rtf']}x RTF{wer_str}, {res['detected_lang']})")
                except Exception as e:
                    print(f"❌ Failed: {e}")

        md_path = WORKSPACE_ROOT / "docs" / "features" / "core" / f"benchmark-{Path(sample_path).stem}.md"
        generate_markdown_report(sample_results, s["name"], md_path)

    save_master_results(all_results)
    print(f"\n💾 Saved all benchmark runs to {BENCHMARK_RESULTS_PATH}")


if __name__ == "__main__":
    main()
