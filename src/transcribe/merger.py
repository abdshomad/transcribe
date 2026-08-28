"""Continuous sequential transcription merger and timeline stitcher."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .models import (
    DiarizedSegment,
    TranscriptSegment,
    TranscriptionResult,
    WordInfo,
)
from .exporters import (
    export_json,
    export_srt,
    export_txt,
    export_vtt,
    export_md,
)


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    mins = int(seconds // 60)
    secs = seconds % 60
    if mins >= 60:
        hrs = mins // 60
        mins = mins % 60
        return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"
    return f"{mins:02d}:{secs:05.2f}"


def extract_sequence_key(filename: str) -> Tuple[int, str]:
    """Extract sequence order from 'Recording 1', 'Part 2', '#3', or trailing numbers."""
    patterns = [
        r'(?:recording|part|track|seq|vol|cd|disk|segment|session)[_ -]*(\d+)',
        r'#\s*(\d+)',
        r'_(\d+)(?:\.[a-zA-Z0-9]+)?$',
        r'(\d+)',
    ]
    for pat in patterns:
        if match := re.search(pat, filename, re.IGNORECASE):
            try:
                return (int(match.group(1)), filename.lower())
            except ValueError:
                pass
    return (999999, filename.lower())


def sort_media_files_by_sequence(media_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort media file entries in natural sequence order."""
    return sorted(media_files, key=lambda f: extract_sequence_key(f.get('name', '')))


def _offset_segment_words(words: Optional[List[WordInfo]], offset: float) -> Optional[List[WordInfo]]:
    """Offset individual word timestamps by cumulative seconds."""
    if not words:
        return words
    return [
        WordInfo(
            word=w.word,
            start=round(w.start + offset, 2),
            end=round(w.end + offset, 2),
            probability=w.probability,
            speaker=w.speaker,
        )
        for w in words
    ]


def _shift_segments(
    segments: List[DiarizedSegment],
    offset: float,
    part_idx: int,
    part_name: str,
    start_id: int = 0,
) -> List[DiarizedSegment]:
    """Shift diarized segment timestamps by offset."""
    shifted: List[DiarizedSegment] = []
    for idx, s in enumerate(segments):
        shifted.append(DiarizedSegment(
            id=start_id + idx,
            speaker=s.speaker,
            start=round(s.start + offset, 2),
            end=round(s.end + offset, 2),
            text=s.text,
            words=_offset_segment_words(s.words, offset),
            emotion=s.emotion,
            events=s.events,
        ))
    return shifted


def combine_transcription_results(
    results_with_metadata: List[Tuple[str, TranscriptionResult]],
    folder_name: str,
) -> TranscriptionResult:
    """Combine multiple sequential transcription results into a single continuous timeline."""
    if not results_with_metadata:
        raise ValueError("No transcription results provided to combine.")

    all_segments: List[DiarizedSegment] = []
    cumulative_offset = 0.0
    languages_used: List[str] = []

    for idx, (part_name, res) in enumerate(results_with_metadata, start=1):
        # Shift segments
        shifted_segs = _shift_segments(
            res.segments,
            cumulative_offset,
            idx,
            part_name,
            start_id=len(all_segments),
        )
        all_segments.extend(shifted_segs)

        # Accumulate metrics
        cumulative_offset += res.duration
        if res.language and res.language not in languages_used:
            languages_used.append(res.language)

    # Calculate combined speakers
    distinct_speakers = sorted(list(set(
        s.speaker for s in all_segments if s.speaker
    )))

    return TranscriptionResult(
        language=", ".join(languages_used) if languages_used else "id",
        duration=round(cumulative_offset, 2),
        segments=all_segments,
        speakers=distinct_speakers,
    )


def export_markdown_combined(
    result: TranscriptionResult,
    output_path: Path,
    title: str,
) -> Path:
    """Export rich formatted Markdown for combined continuous transcript."""
    total_dur_str = format_timestamp(result.duration)
    speakers_str = ", ".join(result.speakers) if result.speakers else "None"
    lines = [
        f"# 🎙️ {title} (Combined Continuous Transcript)",
        "",
        f"> **Total Duration**: `{total_dur_str}` ({result.duration:.1f}s)  ",
        f"> **Language**: `{result.language}`  ",
        f"> **Speakers**: `{len(result.speakers)}` ({speakers_str})  ",
        "",
        "---",
        "",
    ]

    for seg in result.segments:
        spk_tag = f"**[{seg.speaker}]** " if seg.speaker else ""
        ts = f"`{format_timestamp(seg.start)} ➜ {format_timestamp(seg.end)}`"
        lines.append(f"{ts} {spk_tag}{seg.text}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_all_combined_formats(
    result: TranscriptionResult,
    output_dir: Path,
    stem: str,
    formats: List[str],
    title: str,
) -> Dict[str, Path]:
    """Export combined transcription to requested file formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    for fmt in formats:
        p = output_dir / f"{stem}.{fmt}"
        if fmt == "json":
            export_json(result, p)
        elif fmt == "srt":
            export_srt(result, p)
        elif fmt == "vtt":
            export_vtt(result, p)
        elif fmt == "txt":
            export_txt(result, p)
        elif fmt in ("md", "markdown"):
            export_markdown_combined(result, p, title)
        generated[fmt] = p

    return generated
