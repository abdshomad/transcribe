from pathlib import Path
from transcribe.pipeline import AudioTranscriptionPipeline


def test_audio_to_text_pipeline_e2e():
    sample_wav = Path("data/sample/proklamasi.wav")
    assert sample_wav.exists(), "Sample audio file missing"

    pipeline = AudioTranscriptionPipeline(
        whisper_model_size="tiny",
        device="auto",
        enable_diarization=False,
    )

    streamed_segments = []

    def on_seg(seg):
        streamed_segments.append(seg)

    result = pipeline.process(
        sample_wav,
        language="id",
        on_segment=on_seg,
    )

    assert result.duration > 40.0
    assert result.language == "id"
    assert len(result.segments) >= 3
    assert len(streamed_segments) == len(result.segments)

    for seg in result.segments:
        assert seg.start < seg.end
        assert len(seg.text.strip()) > 0
        assert len(seg.words) > 0

    full_text_lower = result.full_text.lower()
    assert "indonesia" in full_text_lower
    assert "agustus" in full_text_lower or "jakarta" in full_text_lower


def test_mp3_format_ingestion():
    sample_mp3 = Path("data/sample/proklamasi.mp3")
    assert sample_mp3.exists(), "Sample MP3 file missing"

    pipeline = AudioTranscriptionPipeline(
        whisper_model_size="tiny",
        device="auto",
        enable_diarization=False,
    )

    result = pipeline.process(sample_mp3, language="id")
    assert result.duration > 40.0
    assert "indonesia" in result.full_text.lower()
