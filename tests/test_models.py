from transcribe.models import (
    WordInfo,
    TranscriptSegment,
    SpeakerSegment,
    DiarizedSegment,
    TranscriptionResult,
)


def test_models_creation():
    word = WordInfo(word="Hello", start=0.0, end=0.5, speaker="SPEAKER_00")
    assert word.word == "Hello"
    assert word.speaker == "SPEAKER_00"

    seg = DiarizedSegment(
        id=1,
        speaker="SPEAKER_01",
        start=0.0,
        end=1.5,
        text="Hello world",
        words=[word],
    )
    assert seg.speaker == "SPEAKER_01"

    result = TranscriptionResult(
        language="en",
        duration=1.5,
        segments=[seg],
        speakers=["SPEAKER_01"],
    )
    assert "SPEAKER_01" in result.full_text
    assert "Hello world" in result.full_text
