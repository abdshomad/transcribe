"""Universal anonymous Google Drive and HTTP audio downloader with progress tracking."""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional
import gdown
import requests

CACHE_DIR = Path("data/downloads")
CACHE_MAX_AGE_SECONDS = 86400  # 24 hours


def extract_gdrive_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from URL."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
        r"/uc\?id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        if match := re.search(pattern, url):
            return match.group(1)
    return None


def is_url(path_or_url: str) -> bool:
    """Check if input string is an HTTP/HTTPS or GDrive URL."""
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def _get_cache_path(url: str) -> Path:
    """Generate cached file path based on URL hash or GDrive ID."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if gdrive_id := extract_gdrive_id(url):
        key = f"gdrive_{gdrive_id}"
    else:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.audio"


def cleanup_expired_cache(max_age: int = CACHE_MAX_AGE_SECONDS) -> None:
    """Remove cached files older than max_age."""
    if not CACHE_DIR.exists():
        return
    now = time.time()
    for item in CACHE_DIR.glob("*"):
        if item.is_file() and (now - item.stat().st_mtime) > max_age:
            try:
                item.unlink()
            except OSError:
                pass


def download_gdrive_anonymous(
    file_id: str,
    dest_path: Path,
    on_progress: Optional[Callable[[int, Optional[int], float], None]] = None,
) -> Path:
    """Download public Google Drive file with progress reporting."""
    temp_dest = dest_path.with_suffix(".tmp")
    if temp_dest.exists():
        temp_dest.unlink()

    # Strategy 1: gdown with direct ID
    output = gdown.download(
        id=file_id,
        output=str(temp_dest),
        quiet=False,
    )
    if output and Path(output).exists() and Path(output).stat().st_size > 0:
        total = Path(output).stat().st_size
        if on_progress:
            on_progress(total, total, 100.0)
        Path(output).rename(dest_path)
        return dest_path

    # Strategy 2: Fallback with full drive URL
    url = f"https://drive.google.com/uc?id={file_id}"
    output = gdown.download(
        url=url,
        output=str(temp_dest),
        quiet=False,
    )
    if output and Path(output).exists() and Path(output).stat().st_size > 0:
        total = Path(output).stat().st_size
        if on_progress:
            on_progress(total, total, 100.0)
        Path(output).rename(dest_path)
        return dest_path

    raise RuntimeError(f"Failed to download Google Drive file: {file_id}. Ensure file is shared publicly ('Anyone with link').")


def download_url(
    url: str,
    use_cache: bool = True,
    on_progress: Optional[Callable[[int, Optional[int], float], None]] = None,
) -> str:
    """Download URL with smart caching and live progress reporting."""
    dest_path = _get_cache_path(url)
    if use_cache and dest_path.exists() and dest_path.stat().st_size > 0:
        sz = dest_path.stat().st_size
        if on_progress:
            on_progress(sz, sz, 100.0)
        return str(dest_path)

    cleanup_expired_cache()

    if gdrive_id := extract_gdrive_id(url):
        download_gdrive_anonymous(gdrive_id, dest_path, on_progress=on_progress)
    else:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; STTAgent/1.0)"}
        temp_dest = dest_path.with_suffix(".tmp")
        start_time = time.time()
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            downloaded = 0
            with open(temp_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if on_progress:
                            elapsed = max(0.001, time.time() - start_time)
                            speed_mb = (downloaded / (1024 * 1024)) / elapsed
                            pct = (downloaded / total * 100.0) if total else 0.0
                            on_progress(downloaded, total, pct, speed_mb)
        temp_dest.rename(dest_path)

    return str(dest_path)
