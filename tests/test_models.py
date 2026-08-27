from transcribe.models import (
    WordInfo,
    TranscriptSegment,
    SpeakerSegment,
    DiarizedSegment,
    TranscriptionResult,
    MODEL_CATALOG,
    CLOUD_MODEL_CATALOG,
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


def test_model_catalogs_and_status_flags():
    """Verify local catalog has implemented=True and cloud catalog has implemented=False."""
    assert len(MODEL_CATALOG) > 0
    assert all(m.implemented is True and m.is_local is True for m in MODEL_CATALOG)

    assert len(CLOUD_MODEL_CATALOG) > 0
    assert all(m.implemented is False and m.is_local is False for m in CLOUD_MODEL_CATALOG)
    cloud_names = [m.name for m in CLOUD_MODEL_CATALOG]
    assert "openai-whisper-api" in cloud_names
    assert "azure-speech-to-text" in cloud_names
    assert "groq-whisper-cloud" in cloud_names
