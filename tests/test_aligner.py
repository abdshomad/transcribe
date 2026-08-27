from transcribe.models import TranscriptSegment, SpeakerSegment, WordInfo
from transcribe.aligner import (
    calculate_overlap,
    assign_speaker_to_word,
    align_transcription_and_diarization,
    format_speaker_label,
)


def test_calculate_overlap():
    assert calculate_overlap(0.0, 2.0, 1.0, 3.0) == 1.0
    assert calculate_overlap(0.0, 1.0, 2.0, 3.0) == 0.0
    assert calculate_overlap(1.0, 4.0, 2.0, 3.0) == 1.0


def test_format_speaker_label():
    m = {}
    assert format_speaker_label("SPEAKER_00", m) == "Speaker 1"
    assert format_speaker_label("SPEAKER_01", m) == "Speaker 2"
    assert format_speaker_label("0", m) == "Speaker 1"


def test_assign_speaker_to_word():
    speakers = [
        SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        SpeakerSegment(speaker="SPEAKER_01", start=2.0, end=5.0),
    ]
    w1 = WordInfo(word="Hi", start=0.5, end=1.0)
    w2 = WordInfo(word="There", start=2.5, end=3.0)
    m = {}

    assert assign_speaker_to_word(w1, speakers, m) == "Speaker 1"
    assert assign_speaker_to_word(w2, speakers, m) == "Speaker 2"


def test_align_transcription_and_diarization():
    raw_segments = [
        TranscriptSegment(
            id=0,
            start=0.0,
            end=1.8,
            text="Welcome to the show.",
            words=[
                WordInfo(word="Welcome", start=0.0, end=0.5),
                WordInfo(word="to", start=0.5, end=0.7),
                WordInfo(word="the", start=0.7, end=0.9),
                WordInfo(word="show.", start=0.9, end=1.5),
            ],
        ),
        TranscriptSegment(
            id=1,
            start=2.2,
            end=4.0,
            text="Thanks for having me.",
            words=[
                WordInfo(word="Thanks", start=2.2, end=2.8),
                WordInfo(word="for", start=2.8, end=3.0),
                WordInfo(word="having", start=3.0, end=3.5),
                WordInfo(word="me.", start=3.5, end=4.0),
            ],
        ),
    ]

    speaker_segments = [
        SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        SpeakerSegment(speaker="SPEAKER_01", start=2.0, end=5.0),
    ]

    aligned = align_transcription_and_diarization(raw_segments, speaker_segments)
    assert len(aligned) == 2
    assert aligned[0].speaker == "Speaker 1"
    assert aligned[1].speaker == "Speaker 2"
    assert aligned[0].words[0].speaker == "Speaker 1"
    assert aligned[1].words[0].speaker == "Speaker 2"
