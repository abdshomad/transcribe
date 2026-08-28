"""Unit tests for continuous sequential transcription merger."""

import pytest
from pathlib import Path
from transcribe.models import (
    DiarizedSegment,
    TranscriptSegment,
    TranscriptionResult,
    WordInfo,
)
from transcribe.merger import (
    extract_sequence_key,
    sort_media_files_by_sequence,
    combine_transcription_results,
    export_all_combined_formats,
)


def test_extract_sequence_key():
    """Verify sequence number extraction across diverse naming patterns."""
    assert extract_sequence_key('Meeting - Recording 1.m4a')[0] == 1
    assert extract_sequence_key('Meeting - Recording 2')[0] == 2
    assert extract_sequence_key('Session_part_03.wav')[0] == 3
    assert extract_sequence_key('Track 4.mp3')[0] == 4
    assert extract_sequence_key('Interview #5.aac')[0] == 5
    assert extract_sequence_key('audio_06.ogg')[0] == 6
    assert extract_sequence_key('random_file_without_number.wav')[0] == 999999


def test_sort_media_files_by_sequence():
    """Verify natural sequence ordering of files."""
    unordered = [
        {'name': 'Meeting - Recording 3', 'id': '3'},
        {'name': 'Meeting - Recording 1.m4a', 'id': '1'},
        {'name': 'Meeting - Recording 2', 'id': '2'},
        {'name': 'Meeting - Recording 10', 'id': '10'},
    ]
    sorted_files = sort_media_files_by_sequence(unordered)
    assert [f['id'] for f in sorted_files] == ['1', '2', '3', '10']


def test_combine_transcription_results():
    """Verify progressive timestamp offset accumulation and section generation."""
    res1 = TranscriptionResult(
        language='id',
        duration=15.0,
        segments=[
            DiarizedSegment(
                id=0,
                speaker='SPEAKER_00',
                start=0.0,
                end=10.0,
                text='Hello world from part one.',
                words=[WordInfo(word='Hello', start=0.0, end=1.0, probability=0.95)],
            )
        ],
        speakers=['SPEAKER_00'],
    )

    res2 = TranscriptionResult(
        language='id',
        duration=20.0,
        segments=[
            DiarizedSegment(
                id=0,
                speaker='SPEAKER_01',
                start=2.0,
                end=8.0,
                text='Welcome to part two.',
                words=[WordInfo(word='Welcome', start=2.0, end=3.0, probability=0.98)],
            )
        ],
        speakers=['SPEAKER_01'],
    )

    combined = combine_transcription_results([('Part 1', res1), ('Part 2', res2)], 'Test Folder')

    # Total duration should be 15.0 + 20.0 = 35.0s
    assert combined.duration == 35.0
    assert len(combined.segments) == 2

    # First segment stays at 0.0 - 10.0
    assert combined.segments[0].start == 0.0
    assert combined.segments[0].end == 10.0
    assert combined.segments[0].words[0].start == 0.0

    # Second segment is shifted by 15.0s -> 17.0 - 23.0
    assert combined.segments[1].start == 17.0
    assert combined.segments[1].end == 23.0
    assert combined.segments[1].words[0].start == 17.0

    # Distinct speakers accumulated
    assert set(combined.speakers) == {'SPEAKER_00', 'SPEAKER_01'}


def test_export_all_combined_formats(tmp_path):
    """Verify file generation across all formats."""
    res = TranscriptionResult(
        language='id',
        duration=5.0,
        segments=[DiarizedSegment(id=0, speaker='SPEAKER_00', start=0.0, end=5.0, text='Sample text')],
        speakers=['SPEAKER_00'],
    )

    exported = export_all_combined_formats(
        result=res,
        output_dir=tmp_path,
        stem='test_combined',
        formats=['json', 'srt', 'vtt', 'txt', 'md'],
        title='Sample Meeting',
    )

    for fmt in ['json', 'srt', 'vtt', 'txt', 'md']:
        assert fmt in exported
        assert Path(exported[fmt]).exists()
        assert Path(exported[fmt]).stat().st_size > 0
