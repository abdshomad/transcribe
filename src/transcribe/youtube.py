"""YouTube subtitle and transcript extractor using yt-dlp.
Fast-path zero-ASR subtitle extraction with graceful fallback message.
"""

import os
import re
import json
import glob
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .models import TranscriptSegment, WordInfo

YOUTUBE_URL_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)([\w-]+)"
)


def is_youtube_url(url: str) -> bool:
    """Check if a string is a valid YouTube URL."""
    if not isinstance(url, str):
        return False
    return bool(YOUTUBE_URL_REGEX.search(url.strip()))


def parse_vtt_timestamp(ts_str: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to total seconds."""
    ts_str = ts_str.strip()
    parts = ts_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return 0.0


def _flush_vtt_segment(
    start: Optional[float],
    end: Optional[float],
    text_lines: List[str],
    seen_text: set,
    seg_id: int,
    segments: List[TranscriptSegment],
) -> int:
    """Helper to clean and flush a single VTT segment."""
    if start is None or not text_lines:
        return seg_id
    raw_text = " ".join(text_lines).strip()
    clean_t = re.sub(r"<[^>]+>", "", raw_text).strip()
    if clean_t and clean_t not in seen_text:
        seen_text.add(clean_t)
        segments.append(
            TranscriptSegment(
                id=seg_id,
                start=start,
                end=end if end is not None else start,
                text=clean_t,
                words=[],
            )
        )
        return seg_id + 1
    return seg_id


def parse_vtt_content(vtt_text: str) -> Tuple[List[TranscriptSegment], str]:
    """Parse VTT subtitle content into clean timestamped TranscriptSegments."""
    lines = vtt_text.splitlines()
    segments: List[TranscriptSegment] = []
    current_start, current_end = None, None
    current_text_lines: List[str] = []
    seg_id = 0
    seen_text = set()
    time_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    )

    for line in lines:
        line = line.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue

        m = time_pattern.search(line)
        if m:
            seg_id = _flush_vtt_segment(current_start, current_end, current_text_lines, seen_text, seg_id, segments)
            current_start = parse_vtt_timestamp(m.group(1))
            current_end = parse_vtt_timestamp(m.group(2))
            current_text_lines = []
        else:
            clean_l = re.sub(r"<[^>]+>", "", line).strip()
            if clean_l and not clean_l.isdigit():
                current_text_lines.append(clean_l)

    _flush_vtt_segment(current_start, current_end, current_text_lines, seen_text, seg_id, segments)
    full_text = " ".join(s.text for s in segments).strip()
    return segments, full_text


def _select_subtitle_file(
    vtt_files: List[str],
    srt_files: List[str],
    preferred_languages: List[str],
) -> Tuple[Optional[str], str]:
    """Select the best matching subtitle file and determine its language."""
    if vtt_files:
        for lang in preferred_languages:
            matching = [f for f in vtt_files if f".{lang}." in f or f.endswith(f".{lang}.vtt")]
            if matching:
                return matching[0], lang.split("-")[0]
        return vtt_files[0], "unknown"
    if srt_files:
        return srt_files[0], "en"
    return None, "unknown"


def fetch_youtube_transcript(
    url: str,
    output_dir: Optional[Path] = None,
    preferred_languages: List[str] = ["en", "en-orig", "id"],
) -> Optional[Dict[str, Any]]:
    """Fetch existing YouTube subtitles/captions using yt-dlp without downloading media."""
    if not is_youtube_url(url):
        return None

    cache_dir = output_dir or (Path("data/downloads/youtube"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = str(Path(tmpdir) / "%(id)s.%(ext)s")
        lang_str = ",".join(preferred_languages)

        cmd = [
            "yt-dlp",
            "--no-update",
            "--skip-download",
            "--write-info-json",
            "--write-auto-sub",
            "--write-subs",
            "--sub-lang", lang_str,
            "--sub-format", "vtt/srt/json3",
            "--ignore-errors",
            "--output", out_template,
            url.strip(),
        ]
        subprocess.run(cmd, capture_output=True, text=True)

        info_files = glob.glob(str(Path(tmpdir) / "*.info.json"))
        if not info_files:
            return None

        with open(info_files[0], "r", encoding="utf-8") as f:
            info_data = json.load(f)

        vtt_files = glob.glob(str(Path(tmpdir) / "*.vtt"))
        srt_files = glob.glob(str(Path(tmpdir) / "*.srt"))
        selected_sub_file, detected_lang = _select_subtitle_file(vtt_files, srt_files, preferred_languages)

        if not selected_sub_file or not os.path.exists(selected_sub_file):
            return None

        with open(selected_sub_file, "r", encoding="utf-8", errors="ignore") as f:
            vtt_content = f.read()

        segments, full_text = parse_vtt_content(vtt_content)
        if not full_text or not segments:
            return None

        return {
            "source": "youtube_subtitles",
            "video_id": info_data.get("id", ""),
            "title": info_data.get("title", "YouTube Video"),
            "url": url,
            "duration": float(info_data.get("duration", 0.0)),
            "language": detected_lang,
            "language_probability": 1.0,
            "segments": segments,
            "full_text": full_text,
            "model": "youtube-captions",
            "speakers": ["SPEAKER_00"],
            "status": "completed"
        }
