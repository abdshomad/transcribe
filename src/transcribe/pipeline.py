"""Unified audio transcription and speaker diarization pipeline with fine-grained stages."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional
from .models import TranscriptionResult, TranscriptSegment
from .audio import get_audio_info
from .downloader import is_url, download_url
from .transcriber import FasterWhisperTranscriber
from .diarizer import PyAnnoteDiarizer
from .aligner import align_transcription_and_diarization
from .exporters import export_json, export_srt, export_vtt, export_txt, export_md


def convert_slice_to_wav_16k(input_path: str, output_path: str, start_offset: float = 0.0) -> str:
    """Convert audio to 16kHz mono WAV, optionally seeking from start_offset."""
    cmd = ["ffmpeg", "-y"]
    if start_offset > 0:
        cmd.extend(["-ss", str(start_offset)])
    cmd.extend([
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ])
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {res.stderr.decode('utf-8', errors='ignore')}")
    return output_path


class AudioTranscriptionPipeline:
    """End-to-end orchestrator for audio transcription + speaker diarization."""

    def __init__(
        self,
        whisper_model_size: str = "base",
        hf_token: Optional[str] = None,
        device: str = "auto",
        enable_diarization: bool = True,
    ):
        self.device = device
        self.enable_diarization = enable_diarization
        self.transcriber = FasterWhisperTranscriber(
            model_size=whisper_model_size,
            device=device,
        )
        self.diarizer = (
            PyAnnoteDiarizer(hf_token=hf_token, device=device)
            if enable_diarization
            else None
        )

    def process(
        self,
        audio_path_or_url: str | Path,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        start_offset: float = 0.0,
        existing_segments: Optional[List[TranscriptSegment]] = None,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> TranscriptionResult:
        """Run complete transcription + diarization with fine-grained progress feedback."""
        source_str = str(audio_path_or_url)
        if is_url(source_str):
            def dl_cb(downloaded, total, pct, speed_mb=0.0):
                if on_progress:
                    on_progress({
                        "stage": "downloading",
                        "downloaded": downloaded,
                        "total": total,
                        "percent": pct,
                        "speed": speed_mb,
                    })

            local_audio_path = download_url(source_str, on_progress=dl_cb)
        else:
            local_audio_path = source_str

        with tempfile.TemporaryDirectory() as tmpdir:
            if on_progress:
                on_progress({
                    "stage": "audio_prep",
                    "message": "⚙️ Normalizing audio (16kHz PCM WAV)...",
                })

            temp_full_wav = os.path.join(tmpdir, "full_16k.wav")
            convert_slice_to_wav_16k(local_audio_path, temp_full_wav, start_offset=0.0)
            total_duration, _, _ = get_audio_info(temp_full_wav)

            temp_slice_wav = os.path.join(tmpdir, "slice_16k.wav")
            if start_offset > 0:
                convert_slice_to_wav_16k(local_audio_path, temp_slice_wav, start_offset=start_offset)
            else:
                temp_slice_wav = temp_full_wav

            if on_progress:
                on_progress({
                    "stage": "vad_scan",
                    "duration": total_duration,
                    "resumed_offset": start_offset,
                    "message": "🔍 Scanning voice activity (Silero VAD) & preparing GPU...",
                })

            def seg_wrapper(seg: TranscriptSegment):
                if start_offset > 0:
                    seg.start += start_offset
                    seg.end += start_offset
                    for w in seg.words:
                        w.start += start_offset
                        w.end += start_offset

                if on_segment:
                    on_segment(seg)
                if on_progress:
                    pct = min(100.0, (seg.end / total_duration * 100.0)) if total_duration > 0 else 0.0
                    on_progress({
                        "stage": "transcribing",
                        "current_time": seg.end,
                        "duration": total_duration,
                        "percent": pct,
                        "segment": seg.model_dump(),
                    })

            # 1. ASR transcription
            raw_new_segments, lang, lang_prob = self.transcriber.transcribe(
                temp_slice_wav,
                language=language,
                on_segment=seg_wrapper,
            )

            # Re-index merged segments
            all_raw = list(existing_segments or []) + raw_new_segments
            for idx, s in enumerate(all_raw):
                s.id = idx

            # 2. Speaker diarization
            speaker_segments = []
            if self.enable_diarization and self.diarizer:
                try:
                    speaker_segments = self.diarizer.diarize(
                        temp_full_wav,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                except Exception as e:
                    print(f"Warning: Diarization skipped ({e}). Single speaker mode.")

            # 3. Temporal Alignment
            aligned_segments = align_transcription_and_diarization(
                all_raw,
                speaker_segments,
            )

            unique_speakers = sorted(list({s.speaker for s in aligned_segments}))

            return TranscriptionResult(
                language=lang,
                language_probability=lang_prob,
                duration=total_duration,
                segments=aligned_segments,
                speakers=unique_speakers,
            )

    def process_and_export(
        self,
        audio_path_or_url: str | Path,
        output_dir: str | Path,
        formats: Optional[List[str]] = None,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        start_offset: float = 0.0,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> dict[str, str]:
        """Process audio/URL and write requested output format files."""
        formats = formats or ["json", "txt", "srt", "vtt", "md"]
        result = self.process(
            audio_path_or_url,
            language=language,
            num_speakers=num_speakers,
            start_offset=start_offset,
            on_segment=on_segment,
            on_progress=on_progress,
        )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stem = Path(str(audio_path_or_url)).stem or "transcript"

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
