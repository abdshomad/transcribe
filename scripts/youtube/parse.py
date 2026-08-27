#!/usr/bin/env python3
"""
Parse downloaded YouTube metadata and transcripts to analyze
ASR/STT, Diarization, and Voice AI models.
"""

import json
import glob
import sys
from pathlib import Path
from collections import defaultdict

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from transcribe.youtube import parse_vtt_content

DATA_DIR = WORKSPACE_ROOT / "data" / "youtube" / "playlist"

KEYWORDS = {
    "ASR/STT": [
        "nemotron", "vibevoice", "vibe-voice", "moonshine", "voxtral",
        "moss", "sats", "sensevoice", "canary", "parakeet", "mms",
        "conformer", "zipformer", "sherpa", "whisper.cpp", "whisperx",
        "scribe", "amazon transcribe", "deepgram", "inworld", "speech engine",
        "faster-whisper", "distil-whisper", "whisper"
    ],
    "Diarization": [
        "pyannote", "diarization", "speaker", "moss sats", "vibevoice",
        "whisperx", "sortformer", "cam++"
    ],
    "Speech-to-Speech / Realtime Agent": [
        "personaplex", "gpt-realtime", "fun audio chat", "covo-audio",
        "dograh", "livekit", "daily", "modal", "gemini live"
    ],
    "TTS / Voice Cloning": [
        "qwen3-tts", "qwen-tts", "kittentts", "voicebox", "maya-1",
        "kugelaudio", "vibe voice tts", "inworld tts", "openvox",
        "elevenlabs", "funaudio", "cosyvoice", "chatterbox"
    ]
}


def clean_vtt(vtt_text: str) -> str:
    _, full_text = parse_vtt_content(vtt_text)
    return full_text


def _extract_subtitles(base: str) -> str:
    """Find and parse subtitles from local subtitle files."""
    for ext in [".en.vtt", ".en-orig.vtt", ".vtt", ".srt"]:
        sub_files = glob.glob(str(DATA_DIR / f"{base}*{ext}"))
        if sub_files:
            try:
                with open(sub_files[0], "r", encoding="utf-8") as sf:
                    return clean_vtt(sf.read())
            except Exception:
                pass
    return ""


def _match_keywords(
    full_content: str,
    info: dict,
    webpage_url: str,
    uploader: str,
    title: str,
    duration: int,
    model_mentions: dict,
) -> dict:
    """Identify matching voice AI keyword categories."""
    detected = defaultdict(list)
    for cat, kw_list in KEYWORDS.items():
        for kw in kw_list:
            if kw in full_content:
                detected[cat].append(kw)
                model_mentions[kw].append({
                    "video_id": info.get("id"),
                    "title": title,
                    "url": webpage_url,
                    "uploader": uploader,
                    "duration": duration,
                })
    return dict(detected)


def analyze_transcripts():
    info_files = sorted(glob.glob(str(DATA_DIR / "*.info.json")))
    print(f"Found {len(info_files)} metadata files in {DATA_DIR}")

    videos = []
    model_mentions = defaultdict(list)

    for info_path in info_files:
        p = Path(info_path)
        base = p.stem.replace(".info", "")
        with open(p, "r", encoding="utf-8") as f:
            info = json.load(f)

        title = info.get("title", "")
        desc = info.get("description", "")
        tags = info.get("tags", [])
        webpage_url = info.get("webpage_url", f"https://www.youtube.com/watch?v={info.get('id')}")
        duration = info.get("duration", 0)
        uploader = info.get("uploader", "")

        subs_text = _extract_subtitles(base)
        full_content = f"{title}\n{desc}\n{' '.join(tags)}\n{subs_text}".lower()
        detected_categories = _match_keywords(
            full_content, info, webpage_url, uploader, title, duration, model_mentions
        )

        videos.append({
            "id": info.get("id"),
            "title": title,
            "url": webpage_url,
            "uploader": uploader,
            "duration": duration,
            "categories": detected_categories,
            "has_transcript": bool(subs_text),
            "transcript_len": len(subs_text.split()),
            "description_snippet": desc[:300] if desc else "",
        })

    # Summary Report
    print("\n" + "="*70)
    print(f"TOTAL VIDEOS ANALYZED: {len(videos)}")
    print("="*70)
    
    for cat, kw_list in KEYWORDS.items():
        print(f"\n--- {cat} Models / Keywords ---")
        for kw in kw_list:
            v_list = model_mentions.get(kw, [])
            if v_list:
                print(f"  • {kw.upper()} ({len(v_list)} videos)")
                for v in v_list[:3]:
                    print(f"     - [{v['uploader']}] {v['title']} ({v['url']})")

    # Save detailed JSON summary
    summary_path = DATA_DIR / "playlist_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_videos": len(videos),
            "videos": videos,
            "model_mentions": dict(model_mentions)
        }, f, indent=2)
    print(f"\nSaved structured summary to {summary_path}")

if __name__ == "__main__":
    analyze_transcripts()
