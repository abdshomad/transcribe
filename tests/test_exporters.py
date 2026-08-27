from pathlib import Path
import tempfile
from transcribe.models import DiarizedSegment, TranscriptionResult
from transcribe.exporters import export_json, export_srt, export_vtt, export_txt, export_md


def test_exporters():
    seg1 = DiarizedSegment(
        id=1,
        speaker="Speaker 1",
        start=1.5,
        end=3.2,
        text="First utterance.",
        emotion="HAPPY",
        events=["LAUGHTER"],
    )
    seg2 = DiarizedSegment(id=2, speaker="Speaker 2", start=3.5, end=5.8, text="Second response.")
    res = TranscriptionResult(language="en", duration=6.0, segments=[seg1, seg2], speakers=["Speaker 1", "Speaker 2"])

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "out.json"
        srt_path = Path(tmpdir) / "out.srt"
        vtt_path = Path(tmpdir) / "out.vtt"
        txt_path = Path(tmpdir) / "out.txt"
        md_path = Path(tmpdir) / "out.md"

        export_json(res, json_path)
        export_srt(res, srt_path)
        export_vtt(res, vtt_path)
        export_txt(res, txt_path)
        export_md(res, md_path)

        assert json_path.exists()
        assert srt_path.exists()
        assert vtt_path.exists()
        assert txt_path.exists()
        assert md_path.exists()

        assert "00:00:01,500 --> 00:00:03,200" in srt_path.read_text()
        assert "<v Speaker 1>First utterance." in vtt_path.read_text()
        assert "# 🎙️ Audio Transcription Transcript" in md_path.read_text()
        md_content = md_path.read_text()
        assert "> **[00:01 ➜ 00:03] Speaker 1** `[HAPPY]` `[LAUGHTER]`:" in md_content
        assert "[Speaker 1] [HAPPY] [LAUGHTER] (1.50s - 3.20s): First utterance." in res.full_text
