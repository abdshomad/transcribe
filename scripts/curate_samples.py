"""Curate and standardize multilingual and multi-dialect 16kHz WAV test audio suite."""

import json
import subprocess
from pathlib import Path
from typing import Dict, List

SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_SOURCES = [
    {
        "id": "indonesian-proklamasi",
        "lang": "id",
        "name": "Indonesian Proklamasi Speech",
        "source": "data/sample/proklamasi.wav",
        "target": "data/samples/indonesian_proklamasi_16k.wav",
        "ground_truth": "Kami bangsa Indonesia dengan ini menjatakan kemerdekaan Indonesia.",
    },
    {
        "id": "english-jfk",
        "lang": "en",
        "name": "English JFK Historic Speech",
        "source": "data/sample/jfk.wav",
        "target": "data/samples/english_jfk_16k.wav",
        "ground_truth": "And so my fellow Americans ask not what your country can do for you ask what you can do for your country.",
    },
    {
        "id": "chinese-sensevoice",
        "lang": "zh",
        "name": "Chinese Conversational Speech",
        "source": "data/models/sensevoice-small/example/zh.mp3",
        "target": "data/samples/chinese_speech_16k.wav",
        "ground_truth": "你好世界",
    },
    {
        "id": "japanese-sensevoice",
        "lang": "ja",
        "name": "Japanese Speech",
        "source": "data/models/sensevoice-small/example/ja.mp3",
        "target": "data/samples/japanese_speech_16k.wav",
        "ground_truth": "こんにちは",
    },
    {
        "id": "cantonese-sensevoice",
        "lang": "yue",
        "name": "Cantonese Speech",
        "source": "data/models/sensevoice-small/example/yue.mp3",
        "target": "data/samples/cantonese_speech_16k.wav",
        "ground_truth": "早晨",
    },
]


def convert_to_16k_mono(input_path: str, output_path: str) -> bool:
    """Convert audio file to 16kHz mono PCM 16-bit WAV via ffmpeg."""
    if not Path(input_path).exists():
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0


def curate_all_samples() -> Dict[str, Dict]:
    """Process and index all benchmark samples."""
    manifest = {}
    print("🎧 Curating multilingual test audio suite...")
    for item in AUDIO_SOURCES:
        src = item["source"]
        tgt = item["target"]
        if Path(src).exists():
            success = convert_to_16k_mono(src, tgt)
            if success:
                import soundfile as sf
                info = sf.info(tgt)
                manifest[item["id"]] = {
                    "id": item["id"],
                    "language": item["lang"],
                    "name": item["name"],
                    "path": tgt,
                    "duration": round(info.duration, 2),
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "ground_truth": item["ground_truth"],
                }
                print(f"  ✅ [{item['lang'].upper()}] {item['name']} -> {tgt} ({info.duration:.2f}s)")
            else:
                print(f"  ❌ Failed converting {src}")
        else:
            print(f"  ⚠️ Source file not found: {src}")

    manifest_path = SAMPLES_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📄 Saved manifest index to {manifest_path} ({len(manifest)} samples)")
    return manifest


if __name__ == "__main__":
    curate_all_samples()
