"""Audio processing and normalization utilities."""

import os
import subprocess
from pathlib import Path
from pathlib import Path
from typing import Callable, Optional, Tuple
import soundfile as sf


def probe_media_duration(input_path: str | Path) -> Optional[float]:
    """Probe input media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception:
        pass
    return None


def parse_ffmpeg_progress_line(line: str) -> Optional[float]:
    """Parse out_time_us or out_time from FFmpeg progress line."""
    line = line.strip()
    if line.startswith("out_time_us="):
        val = line.split("=", 1)[1]
        try:
            return int(val) / 1_000_000.0
        except ValueError:
            return None
    if line.startswith("out_time="):
        val = line.split("=", 1)[1]
        parts = val.split(":")
        if len(parts) == 3:
            try:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except ValueError:
                return None
    return None


def _build_ffmpeg_cmd(input_path: str, output_path: str, start_offset: float = 0.0) -> list[str]:
    """Build FFmpeg command line with progress streaming."""
    cmd = ["ffmpeg", "-y"]
    if start_offset > 0:
        cmd.extend(["-ss", str(start_offset)])
    cmd.extend([
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-progress", "pipe:1",
        "-nostats",
        output_path,
    ])
    return cmd


def _process_ffmpeg_stream(
    proc: subprocess.Popen,
    total_duration: Optional[float],
    on_progress: Optional[Callable[[dict], None]],
) -> None:
    """Read FFmpeg stdout and trigger conversion progress callbacks."""
    if not proc.stdout:
        return
    for line in proc.stdout:
        cur_time = parse_ffmpeg_progress_line(line)
        if cur_time is None or not on_progress:
            continue
        if total_duration and total_duration > 0:
            pct = min(100.0, (cur_time / total_duration) * 100.0)
            on_progress({
                "stage": "converting",
                "percent": pct,
                "current_time": cur_time,
                "duration": total_duration,
                "message": f"⚙️ Converting media... {int(pct)}%",
            })
        else:
            on_progress({
                "stage": "converting",
                "percent": 0.0,
                "current_time": cur_time,
                "message": f"⚙️ Converting media... ({int(cur_time)}s)",
            })


def convert_to_wav_16k(
    input_path: str | Path,
    output_path: str | Path,
    start_offset: float = 0.0,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> str:
    """Convert any audio/video input to 16kHz mono WAV with progress streaming."""
    input_path = str(input_path)
    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_dur = probe_media_duration(input_path)
    cmd = _build_ffmpeg_cmd(input_path, output_path, start_offset=start_offset)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    _process_ffmpeg_stream(proc, total_dur, on_progress)
    _, stderr_text = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {stderr_text}")

    if on_progress:
        on_progress({
            "stage": "converting",
            "percent": 100.0,
            "duration": total_dur or 0.0,
            "message": "⚙️ Converting media... 100%",
        })
    return output_path


def get_audio_info(audio_path: str | Path) -> Tuple[float, int, int]:
    """Get (duration_seconds, sample_rate, channels) of audio file."""
    with sf.SoundFile(str(audio_path)) as f:
        duration = len(f) / f.samplerate
        return duration, f.samplerate, f.channels
