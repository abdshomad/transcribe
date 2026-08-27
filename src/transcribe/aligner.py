"""Alignment logic matching ASR words and segments to speaker diarization tracks."""

import re
from typing import Dict, List
from .models import TranscriptSegment, SpeakerSegment, DiarizedSegment, WordInfo


def calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate the overlap duration between two time intervals."""
    return max(0.0, min(end1, end2) - max(start1, start2))


def format_speaker_label(raw: str, mapping: Dict[str, str]) -> str:
    """Format raw speaker string into standardized 'Speaker 1, 2, 3...'."""
    if raw in mapping:
        return mapping[raw]

    # Check for SPEAKER_00 / SPEAKER_01 pattern
    if match := re.search(r"speaker_?(\d+)", raw, re.IGNORECASE):
        num = int(match.group(1)) + 1
        label = f"Speaker {num}"
    elif raw.isdigit():
        label = f"Speaker {int(raw) + 1}"
    else:
        label = f"Speaker {len(mapping) + 1}"

    mapping[raw] = label
    return label


def assign_speaker_to_word(
    word: WordInfo,
    speaker_segments: List[SpeakerSegment],
    speaker_map: Dict[str, str],
) -> str:
    """Find the speaker with maximum overlap for a single word."""
    if not speaker_segments:
        return "Speaker 1"

    best_speaker = speaker_segments[0].speaker
    best_overlap = 0.0

    for spk_seg in speaker_segments:
        overlap = calculate_overlap(word.start, word.end, spk_seg.start, spk_seg.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = spk_seg.speaker

    if best_overlap == 0.0:
        midpoint = (word.start + word.end) / 2.0
        closest_seg = min(
            speaker_segments,
            key=lambda s: min(abs(s.start - midpoint), abs(s.end - midpoint)),
        )
        best_speaker = closest_seg.speaker

    return format_speaker_label(best_speaker, speaker_map)


def align_transcription_and_diarization(
    transcript_segments: List[TranscriptSegment],
    speaker_segments: List[SpeakerSegment],
) -> List[DiarizedSegment]:
    """Align raw ASR transcript segments with speaker diarization timeline."""
    speaker_map: Dict[str, str] = {}

    if not speaker_segments:
        return [
            DiarizedSegment(
                id=seg.id,
                speaker="Speaker 1",
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=seg.words,
                emotion=seg.emotion,
                events=seg.events,
            )
            for seg in transcript_segments
        ]

    aligned_segments: List[DiarizedSegment] = []

    for seg in transcript_segments:
        annotated_words: List[WordInfo] = []
        for word in seg.words:
            spk = assign_speaker_to_word(word, speaker_segments, speaker_map)
            annotated_words.append(
                WordInfo(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    probability=word.probability,
                    speaker=spk,
                )
            )

        speaker_scores: dict[str, float] = {}
        for spk_seg in speaker_segments:
            overlap = calculate_overlap(seg.start, seg.end, spk_seg.start, spk_seg.end)
            if overlap > 0:
                speaker_scores[spk_seg.speaker] = speaker_scores.get(spk_seg.speaker, 0.0) + overlap

        if speaker_scores:
            dominant_speaker = max(speaker_scores.items(), key=lambda x: x[1])[0]
        else:
            midpoint = (seg.start + seg.end) / 2.0
            closest_seg = min(
                speaker_segments,
                key=lambda s: min(abs(s.start - midpoint), abs(s.end - midpoint)),
            )
            dominant_speaker = closest_seg.speaker

        norm_speaker = format_speaker_label(dominant_speaker, speaker_map)

        aligned_segments.append(
            DiarizedSegment(
                id=seg.id,
                speaker=norm_speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=annotated_words,
                emotion=seg.emotion,
                events=seg.events,
            )
        )

    return aligned_segments
