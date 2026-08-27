"""Output format exporters for transcription and diarization results."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from .models import TranscriptionResult


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into WebVTT timestamp (HH:MM:SS.mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def _format_natural_duration(seconds: float) -> str:
    """Format seconds into natural unit string (e.g. 2h 41m 02s, 14m 20s, 48s)."""
    sec_num = int(seconds)
    hrs = sec_num // 3600
    mins = (sec_num % 3600) // 60
    secs = sec_num % 60
    if hrs > 0:
        return f"{hrs}h {mins}m {secs:02d}s"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def _format_time_simple(seconds: float) -> str:
    """Format seconds into digital timestamp HH:MM:SS or MM:SS."""
    sec_num = int(seconds)
    hrs = sec_num // 3600
    mins = (sec_num % 3600) // 60
    secs = sec_num % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def export_json(result: TranscriptionResult, output_path: Optional[str | Path] = None) -> str:
    """Export to formatted JSON string or file."""
    data = result.model_dump()
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    if output_path:
        Path(output_path).write_text(json_str, encoding="utf-8")
    return json_str


def export_srt(result: TranscriptionResult, output_path: Optional[str | Path] = None) -> str:
    """Export to SRT subtitle format with speaker attribution."""
    lines = []
    for i, seg in enumerate(result.segments, start=1):
        start_ts = _format_timestamp_srt(seg.start)
        end_ts = _format_timestamp_srt(seg.end)
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"[{seg.speaker}] {seg.text.strip()}")
        lines.append("")

    srt_str = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(srt_str, encoding="utf-8")
    return srt_str


def export_vtt(result: TranscriptionResult, output_path: Optional[str | Path] = None) -> str:
    """Export to WebVTT subtitle format."""
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        start_ts = _format_timestamp_vtt(seg.start)
        end_ts = _format_timestamp_vtt(seg.end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"<v {seg.speaker}>{seg.text.strip()}")
        lines.append("")

    vtt_str = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(vtt_str, encoding="utf-8")
    return vtt_str


def export_txt(result: TranscriptionResult, output_path: Optional[str | Path] = None) -> str:
    """Export to plain text dialogue format."""
    txt_str = result.full_text
    if output_path:
        Path(output_path).write_text(txt_str, encoding="utf-8")
    return txt_str


def export_md(result: TranscriptionResult, output_path: Optional[str | Path] = None) -> str:
    """Export to structured Markdown dialogue with natural duration table."""
    speakers_list = ", ".join(result.speakers) if result.speakers else "Speaker 1"
    nat_dur = _format_natural_duration(result.duration)
    lines = [
        "# 🎙️ Audio Transcription Transcript",
        "",
        "| Property | Details |",
        "| :--- | :--- |",
        f"| **Duration** | {nat_dur} ({result.duration:.2f}s) |",
        f"| **Language** | {result.language.upper()} ({result.language_probability * 100:.0f}%) |",
        f"| **Speakers** | {speakers_list} |",
        f"| **Segments Count** | {len(result.segments)} |",
        "",
        "---",
        "",
        "### 💬 Dialogue Timeline",
        "",
    ]

    for seg in result.segments:
        start_fmt = _format_time_simple(seg.start)
        end_fmt = _format_time_simple(seg.end)
        badge = ""
        if seg.emotion and seg.emotion != "NEUTRAL":
            badge += f" `[{seg.emotion}]`"
        if seg.events:
            badge += f" `[{' '.join(seg.events)}]`"
        lines.append(f"> **[{start_fmt} ➜ {end_fmt}] {seg.speaker}**{badge}:")
        lines.append(f"> {seg.text.strip()}")
        lines.append("")

    md_str = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(md_str, encoding="utf-8")
    return md_str


def export_transcription(
    result: TranscriptionResult,
    output_dir: str | Path,
    stem: str = "transcript",
    formats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Export transcription result to multiple file formats."""
    formats = formats or ["json", "txt", "srt", "vtt", "md"]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exported_files = {}
    for fmt in formats:
        p = out_path / f"{stem}.{fmt}"
        if fmt == "json":
            export_json(result, p)
        elif fmt == "srt":
            export_srt(result, p)
        elif fmt == "vtt":
            export_vtt(result, p)
        elif fmt == "txt":
            export_txt(result, p)
        elif fmt == "md":
            export_md(result, p)
        exported_files[fmt] = str(p)

    return exported_files
