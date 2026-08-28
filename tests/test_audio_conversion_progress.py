"""Unit tests for live media conversion progress and duration probing."""

from pathlib import Path
from transcribe.audio import (
    probe_media_duration,
    parse_ffmpeg_progress_line,
    _build_ffmpeg_cmd,
    convert_to_wav_16k,
    get_audio_info,
)


def test_parse_ffmpeg_progress_line():
    """Verify parsing of various FFmpeg progress line formats."""
    # out_time_us (microseconds)
    assert parse_ffmpeg_progress_line("out_time_us=15000000") == 15.0
    assert parse_ffmpeg_progress_line("out_time_us=0") == 0.0

    # out_time (HH:MM:SS.microsec)
    assert parse_ffmpeg_progress_line("out_time=00:01:30.500000") == 90.5
    assert parse_ffmpeg_progress_line("out_time=01:00:00.000000") == 3600.0

    # Irrelevant lines
    assert parse_ffmpeg_progress_line("progress=continue") is None
    assert parse_ffmpeg_progress_line("fps=24.5") is None
    assert parse_ffmpeg_progress_line("bitrate=128kbits/s") is None


def test_build_ffmpeg_cmd():
    """Verify FFmpeg command line includes -progress pipe:1 and 16kHz settings."""
    cmd = _build_ffmpeg_cmd("input.mp4", "output.wav", start_offset=10.0)
    assert "-progress" in cmd
    assert "pipe:1" in cmd
    assert "-ar" in cmd
    assert "16000" in cmd
    assert "-ss" in cmd
    assert "10.0" in cmd


def test_probe_media_duration_and_conversion_progress(tmp_path):
    """Test duration probing and live conversion progress callbacks on real audio."""
    sample_wav = Path("data/sample/proklamasi.wav")
    if not sample_wav.exists():
        return

    dur = probe_media_duration(sample_wav)
    assert dur is not None
    assert 45.0 <= dur <= 55.0

    output_wav = tmp_path / "converted_16k.wav"
    progress_events = []

    def on_prog(info: dict):
        progress_events.append(info)

    result_path = convert_to_wav_16k(sample_wav, output_wav, on_progress=on_prog)
    assert Path(result_path).exists()
    assert len(progress_events) > 0

    # Verify final event is 100%
    final_event = progress_events[-1]
    assert final_event.get("stage") == "converting"
    assert final_event.get("percent") == 100.0

    # Check output audio properties
    duration, sr, channels = get_audio_info(output_wav)
    assert sr == 16000
    assert channels == 1
    assert duration > 0
