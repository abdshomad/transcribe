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


def parse_vtt_content(vtt_text: str) -> Tuple[List[TranscriptSegment], str]:
    """Parse VTT subtitle content into clean timestamped TranscriptSegments."""
    lines = vtt_text.splitlines()
    segments: List[TranscriptSegment] = []
    
    current_start = None
    current_end = None
    current_text_lines = []
    seg_id = 0
    seen_text = set()
    
    time_pattern = re.compile(r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})")
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
            
        m = time_pattern.search(line)
        if m:
            # Flush previous segment
            if current_start is not None and current_text_lines:
                raw_text = " ".join(current_text_lines).strip()
                clean_t = re.sub(r"<[^>]+>", "", raw_text).strip()
                if clean_t and clean_t not in seen_text:
                    seen_text.add(clean_t)
                    segments.append(
                        TranscriptSegment(
                            id=seg_id,
                            start=current_start,
                            end=current_end,
                            text=clean_t,
                            words=[],
                        )
                    )
                    seg_id += 1
            current_start = parse_vtt_timestamp(m.group(1))
            current_end = parse_vtt_timestamp(m.group(2))
            current_text_lines = []
        else:
            # Regular text or markup line
            clean_l = re.sub(r"<[^>]+>", "", line).strip()
            if clean_l and not clean_l.isdigit():
                current_text_lines.append(clean_l)
                
    # Flush last segment
    if current_start is not None and current_text_lines:
        raw_text = " ".join(current_text_lines).strip()
        clean_t = re.sub(r"<[^>]+>", "", raw_text).strip()
        if clean_t and clean_t not in seen_text:
            segments.append(
                TranscriptSegment(
                    id=seg_id,
                    start=current_start,
                    end=current_end,
                    text=clean_t,
                    words=[],
                )
            )
            
    full_text = " ".join(s.text for s in segments).strip()
    return segments, full_text


def fetch_youtube_transcript(
    url: str,
    output_dir: Optional[Path] = None,
    preferred_languages: List[str] = ["en", "en-orig", "id"],
) -> Optional[Dict[str, Any]]:
    """
    Fetch existing YouTube subtitles/captions using yt-dlp without downloading media.
    
    Returns:
        Dict with video metadata and parsed TranscriptSegments if subtitles exist.
        None if no captions/subtitles are found.
    """
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
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check info json
        info_files = glob.glob(str(Path(tmpdir) / "*.info.json"))
        if not info_files:
            return None
            
        with open(info_files[0], "r", encoding="utf-8") as f:
            info_data = json.load(f)
            
        title = info_data.get("title", "YouTube Video")
        duration = float(info_data.get("duration", 0.0))
        video_id = info_data.get("id", "")
        
        # Check for generated subtitles
        vtt_files = glob.glob(str(Path(tmpdir) / "*.vtt"))
        srt_files = glob.glob(str(Path(tmpdir) / "*.srt"))
        
        selected_sub_file = None
        detected_lang = "unknown"
        
        if vtt_files:
            selected_sub_file = vtt_files[0]
            for lang in preferred_languages:
                matching = [f for f in vtt_files if f".{lang}." in f or f.endswith(f".{lang}.vtt")]
                if matching:
                    selected_sub_file = matching[0]
                    detected_lang = lang.split("-")[0]
                    break
        elif srt_files:
            selected_sub_file = srt_files[0]
            detected_lang = "en"
            
        if not selected_sub_file or not os.path.exists(selected_sub_file):
            return None
            
        with open(selected_sub_file, "r", encoding="utf-8", errors="ignore") as f:
            vtt_content = f.read()
            
        segments, full_text = parse_vtt_content(vtt_content)
        
        if not full_text or not segments:
            return None
            
        return {
            "source": "youtube_subtitles",
            "video_id": video_id,
            "title": title,
            "url": url,
            "duration": duration,
            "language": detected_lang,
            "language_probability": 1.0,
            "segments": segments,
            "full_text": full_text,
            "model": "youtube-captions",
            "speakers": ["SPEAKER_00"],
            "status": "completed"
        }
