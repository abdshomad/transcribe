"""Universal anonymous Google Drive and HTTP audio downloader with progress tracking."""

import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional
import gdown
import requests

CACHE_DIR = Path("data/downloads")
CACHE_MAX_AGE_SECONDS = 86400  # 24 hours


MEDIA_EXTENSIONS = {
    ".m4a", ".wav", ".mp3", ".mp4", ".aac", ".flac", ".ogg", ".webm",
    ".mkv", ".mov", ".wma", ".opus", ".3gp", ".m4v", ".alac", ".aiff"
}
MEDIA_KEYWORDS = ["recording", "meeting", "audio", "voice", "call", "record", "track"]
NON_MEDIA_KEYWORDS = ["notes by gemini", "google doc", "google sheet", "google slide", "google form"]
NON_MEDIA_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".txt", ".md", ".csv", ".json", ".zip", ".tar", ".gz"
}


def is_media_candidate(filename: str) -> bool:
    """Check if a file is an audio/video candidate by extension or recording keyword."""
    name_lower = filename.lower()
    for non_kw in NON_MEDIA_KEYWORDS:
        if non_kw in name_lower:
            return False
    ext = Path(filename).suffix.lower()
    if ext in NON_MEDIA_EXTENSIONS:
        return False
    if ext in MEDIA_EXTENSIONS:
        return True
    return any(kw in name_lower for kw in MEDIA_KEYWORDS)


def extract_gdrive_folder_id(url: str) -> Optional[str]:
    """Extract Google Drive folder ID from URL."""
    patterns = [
        r"/drive/folders/([a-zA-Z0-9_-]+)",
        r"/drive/u/\d+/folders/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        if match := re.search(pattern, url):
            return match.group(1)
    return None


def is_gdrive_folder(url: str) -> bool:
    """Check if URL is a Google Drive folder link."""
    return bool(extract_gdrive_folder_id(url))


def fetch_gdrive_folder_contents(url_or_id: str) -> tuple[str, list[dict[str, str]]]:
    """Fetch folder title and filtered list of media candidate files from a public Drive folder."""
    folder_id = extract_gdrive_folder_id(url_or_id) or url_or_id
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    # Extract folder title
    folder_title = f"gdrive_folder_{folder_id}"
    try:
        r = requests.get(folder_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if title_m := re.search(r"<title>(.*?)( - Google Drive)?</title>", r.text):
            folder_title = title_m.group(1).replace(" - Google Drive", "").strip()
    except Exception:
        pass

    # Retrieve folder items using gdown dry run
    files = gdown.download_folder(url=folder_url, skip_download=True, quiet=True)
    media_files: list[dict[str, str]] = []
    for f in (files or []):
        file_name = Path(f.path).name
        if is_media_candidate(file_name):
            media_files.append({
                "id": str(f.id),
                "name": file_name,
                "path": str(f.path),
                "url": f"https://drive.google.com/file/d/{f.id}/view",
            })
    return folder_title, media_files


def is_google_doc(url: str) -> bool:
    """Check if URL points to a Google Docs, Sheets, Slides, or Forms document."""
    doc_patterns = ["docs.google.com/document", "docs.google.com/spreadsheets", "docs.google.com/presentation", "docs.google.com/forms"]
    return any(p in url for p in doc_patterns)


def extract_gdrive_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from URL."""
    if is_google_doc(url):
        raise ValueError(
            "The provided URL points to a Google Docs/Sheets text document (such as Gemini meeting notes), "
            "not an audio or video recording file. Please provide a Google Drive audio/video link or folder link."
        )
    if is_gdrive_folder(url):
        raise ValueError(
            "The provided URL is a Google Drive folder link, not a direct file link. "
            "Please provide a direct link to an individual audio file (e.g. https://drive.google.com/file/d/<file-id>/view)."
        )
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


def _is_html_content(file_path: Path) -> bool:
    """Inspect beginning of file to check if it's HTML text rather than media."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(512).strip().lower()
            return header.startswith(b"<!doctype html") or header.startswith(b"<html") or b"<head" in header
    except Exception:
        return False


def _get_cache_path(url: str) -> Path:
    """Generate cached file path based on URL hash or GDrive ID."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        gdrive_id = extract_gdrive_id(url)
    except ValueError:
        gdrive_id = None
    if gdrive_id:
        key = f"gdrive_{gdrive_id}"
    else:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.audio"


def cleanup_expired_cache(max_age: int = CACHE_MAX_AGE_SECONDS) -> None:
    """Remove cached files older than max_age or containing corrupt HTML."""
    if not CACHE_DIR.exists():
        return
    now = time.time()
    for item in CACHE_DIR.glob("*"):
        if not item.is_file():
            continue
        is_expired = (now - item.stat().st_mtime) > max_age
        is_corrupt_html = _is_html_content(item)
        if is_expired or is_corrupt_html:
            try:
                item.unlink()
            except OSError:
                pass


def download_gdrive_with_ytdlp(file_id: str, dest_path: Path) -> Optional[Path]:
    """Download Google Drive stream audio using yt-dlp fallback for view-only recordings."""
    url = f"https://drive.google.com/file/d/{file_id}/view"
    temp_dest = dest_path.with_suffix(".tmp")
    if temp_dest.exists():
        try:
            temp_dest.unlink()
        except OSError:
            pass
    ytdlp_bin = shutil.which("yt-dlp") or "yt-dlp"
    cmd = [
        ytdlp_bin,
        "-f", "ba/b",
        "--no-playlist",
        "--no-update",
        "-o", str(temp_dest),
        url,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        if res.returncode == 0:
            if temp_dest.exists() and temp_dest.stat().st_size > 0:
                if dest_path.exists():
                    dest_path.unlink()
                temp_dest.replace(dest_path)
                return dest_path
            for match in dest_path.parent.glob(f"{temp_dest.name}*"):
                if match.is_file() and match.stat().st_size > 0:
                    if dest_path.exists():
                        dest_path.unlink()
                    match.replace(dest_path)
                    return dest_path
    except Exception:
        pass
    return None


def download_gdrive_anonymous(
    file_id: str,
    dest_path: Path,
    on_progress: Optional[Callable[[int, Optional[int], float], None]] = None,
) -> Path:
    """Download public Google Drive file with progress reporting and yt-dlp fallback."""
    temp_dest = dest_path.with_suffix(".tmp")
    if temp_dest.exists():
        try:
            temp_dest.unlink()
        except OSError:
            pass

    # Strategy 1: gdown with direct ID
    try:
        output = gdown.download(
            id=file_id,
            output=str(temp_dest),
            quiet=False,
        )
        if output and Path(output).exists() and Path(output).stat().st_size > 0:
            total = Path(output).stat().st_size
            if on_progress:
                on_progress(total, total, 100.0)
            Path(output).replace(dest_path)
            return dest_path
    except Exception:
        pass

    if temp_dest.exists():
        try:
            temp_dest.unlink()
        except OSError:
            pass

    # Strategy 2: Fallback with full drive URL
    try:
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
            Path(output).replace(dest_path)
            return dest_path
    except Exception:
        pass

    if temp_dest.exists():
        try:
            temp_dest.unlink()
        except OSError:
            pass

    # Strategy 3: Fallback with yt-dlp for view-only recordings
    if ytdlp_res := download_gdrive_with_ytdlp(file_id, dest_path):
        if on_progress:
            sz = ytdlp_res.stat().st_size
            on_progress(sz, sz, 100.0)
        return ytdlp_res

    raise RuntimeError(f"Failed to download Google Drive file: {file_id}. Ensure file is shared publicly ('Anyone with link').")


def _prepare_download_headers(temp_dest: Path) -> tuple[dict[str, str], int]:
    """Prepare request headers and detect existing byte offset for resume."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; STTAgent/1.0)"}
    existing_bytes = temp_dest.stat().st_size if temp_dest.exists() else 0
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
    return headers, existing_bytes


def _write_stream_chunks(
    response: requests.Response,
    file_obj: Any,
    existing_bytes: int,
    is_range: bool,
    total: Optional[int],
    start_time: float,
    on_progress: Optional[Callable[[int, Optional[int], float, float], None]],
) -> None:
    """Stream chunks to file and trigger progress callbacks."""
    downloaded = existing_bytes if is_range else 0
    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue
        file_obj.write(chunk)
        downloaded += len(chunk)
        if on_progress:
            elapsed = max(0.001, time.time() - start_time)
            bytes_diff = downloaded - (existing_bytes if is_range else 0)
            speed_mb = (bytes_diff / (1024 * 1024)) / elapsed
            pct = (downloaded / total * 100.0) if total else 0.0
            on_progress(downloaded, total, pct, speed_mb)


def _download_http_with_resume(
    url: str,
    temp_dest: Path,
    on_progress: Optional[Callable[[int, Optional[int], float, float], None]] = None,
) -> None:
    """Download HTTP resource with Range auto-resume support on partial .tmp files."""
    headers, existing_bytes = _prepare_download_headers(temp_dest)
    start_time = time.time()
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code == 416:
            return
        r.raise_for_status()

        is_range = r.status_code == 206
        content_len = int(r.headers.get("content-length", 0)) or 0
        total = (existing_bytes + content_len) if is_range and content_len else (content_len or None)
        mode = "ab" if is_range else "wb"

        with open(temp_dest, mode) as f:
            _write_stream_chunks(r, f, existing_bytes, is_range, total, start_time, on_progress)


def download_url(
    url: str,
    use_cache: bool = True,
    on_progress: Optional[Callable[[int, Optional[int], float], None]] = None,
) -> str:
    """Download URL with smart caching and live progress reporting."""
    dest_path = _get_cache_path(url)
    if use_cache and dest_path.exists() and dest_path.stat().st_size > 0:
        if _is_html_content(dest_path):
            dest_path.unlink()
        else:
            sz = dest_path.stat().st_size
            if on_progress:
                on_progress(sz, sz, 100.0)
            return str(dest_path)

    cleanup_expired_cache()

    if is_google_doc(url):
        raise ValueError(
            "The provided URL points to a Google Docs/Sheets text document (such as Gemini meeting notes), "
            "not an audio or video recording file. Please provide a Google Drive audio/video link or folder link."
        )

    if is_gdrive_folder(url):
        raise ValueError(
            "The provided URL is a Google Drive folder link, not an individual audio file. "
            "To transcribe all files in this folder, use batch processing (`transcribe run <url>`)."
        )

    if gdrive_id := extract_gdrive_id(url):
        download_gdrive_anonymous(gdrive_id, dest_path, on_progress=on_progress)
    else:
        temp_dest = dest_path.with_suffix(".tmp")
        _download_http_with_resume(url, temp_dest, on_progress=on_progress)
        temp_dest.rename(dest_path)
        if _is_html_content(dest_path):
            dest_path.unlink()
            raise RuntimeError(
                f"Failed to download audio from '{url}': Received HTML page instead of audio stream. "
                "If using Google Drive, make sure the link points to a single file with public ('Anyone with the link') access."
            )

    return str(dest_path)
