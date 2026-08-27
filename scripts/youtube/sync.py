#!/usr/bin/env python3
"""
Incremental YouTube Playlist Downloader & Subtitle Sync

Features:
- Fetches flat playlist entries via yt-dlp.
- Pre-checks existing `*.info.json` files in `data/youtube/playlist/` to skip already downloaded videos.
- Downloads metadata (.info.json) and subtitles (.vtt / .srt) ONLY for missing items.
- Automatically triggers `scripts/youtube/parse.py` to refresh intelligence summary.
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = WORKSPACE_ROOT / "data" / "youtube" / "playlist"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLIsFyaUd-B31sk5X4_XvjvqApKUGvkTAA"

def get_existing_video_ids(directory: Path) -> set:
    """Scan directory for existing video IDs from info.json files."""
    existing_ids = set()
    for file_path in directory.glob("*.info.json"):
        name = file_path.stem.replace(".info", "")
        parts = name.split("-", 1)
        if len(parts) == 2:
            existing_ids.add(parts[1])
        else:
            existing_ids.add(parts[0])
    return existing_ids

def fetch_playlist_entries(url: str):
    """Fetch flat metadata of all videos in the playlist."""
    print(f"📡 Fetching playlist index from: {url}")
    cmd = [
        "yt-dlp",
        "--no-update",
        "--flat-playlist",
        "--dump-single-json",
        url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    entries = data.get("entries", [])
    title = data.get("title", "Voice AI Playlist")
    return title, entries

def download_video_assets(video_id: str, playlist_index: int, output_dir: Path):
    """Download info.json and subtitles for a single video."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    out_template = str(output_dir / f"{playlist_index:03d}-{video_id}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "--no-update",
        "--skip-download",
        "--write-info-json",
        "--write-auto-sub",
        "--write-subs",
        "--sub-lang", "en,en-orig,id",
        "--sub-format", "vtt/json3/srt",
        "--ignore-errors",
        "--output", out_template,
        video_url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0

def sync_playlist():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Check existing downloads
    existing_ids = get_existing_video_ids(DATA_DIR)
    print(f"📂 Found {len(existing_ids)} previously downloaded videos in {DATA_DIR}")
    
    # 2. Fetch current playlist state
    try:
        title, entries = fetch_playlist_entries(PLAYLIST_URL)
    except Exception as e:
        print(f"❌ Failed to fetch playlist index: {e}")
        return
        
    print(f"📋 Playlist '{title}' has {len(entries)} total entries.")
    
    # 3. Identify missing items
    missing_entries = []
    for idx, entry in enumerate(entries, start=1):
        vid_id = entry.get("id")
        if vid_id and vid_id not in existing_ids:
            missing_entries.append((idx, vid_id, entry.get("title", "Unknown Title")))
            
    if not missing_entries:
        print("✅ All playlist items are already downloaded! Nothing to do.")
    else:
        print(f"🚀 Found {len(missing_entries)} new/missing videos to download:")
        for idx, vid_id, vid_title in missing_entries:
            print(f"   [{idx:03d}] {vid_id} - {vid_title}")
            success = download_video_assets(vid_id, idx, DATA_DIR)
            if success:
                print(f"      ↳ Downloaded metadata & subtitles.")
            else:
                print(f"      ⚠️ Partial download or error.")
                
    # 4. Trigger auto-parsing if needed
    summary_file = DATA_DIR / "playlist_analysis_summary.json"
    if missing_entries or not summary_file.exists():
        print("\n🔄 Running analysis parser to update model intelligence summary...")
        parser_script = Path(__file__).resolve().parent / "parse.py"
        if parser_script.exists():
            subprocess.run([sys.executable, str(parser_script)], check=False)
        else:
            print("⚠️ Parser script not found.")
    else:
        print(f"ℹ️ Summary `{summary_file}` is already up-to-date.")

if __name__ == "__main__":
    sync_playlist()
