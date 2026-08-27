"""Audio processing and normalization utilities."""

import os
import subprocess
from pathlib import Path
from typing import Tuple
import soundfile as sf


def convert_to_wav_16k(input_path: str | Path, output_path: str | Path) -> str:
    """Convert any audio/video input to 16kHz mono WAV using ffmpeg."""
    input_path = str(input_path)
    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.decode('utf-8', errors='ignore')}")
    return output_path


def get_audio_info(audio_path: str | Path) -> Tuple[float, int, int]:
    """Get (duration_seconds, sample_rate, channels) of audio file."""
    with sf.SoundFile(str(audio_path)) as f:
        duration = len(f) / f.samplerate
        return duration, f.samplerate, f.channels
