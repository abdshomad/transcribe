"""Speaker diarization engine using PyAnnote or local energy clustering."""

import os
from pathlib import Path
from typing import List, Optional
from .models import SpeakerSegment


def _read_hf_token() -> Optional[str]:
    """Look for HF token in environment or local secrets file."""
    if token := os.getenv("HF_TOKEN"):
        return token
    secrets_path = Path(".secrets")
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class PyAnnoteDiarizer:
    """PyAnnote Speaker Diarization pipeline wrapper."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
    ):
        self.hf_token = hf_token or _read_hf_token()
        self.model_name = model_name

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load pyannote pipeline with compatibility for token/use_auth_token."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from pyannote.audio import Pipeline

            try:
                self._pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=self.hf_token,
                )
            except (TypeError, ValueError):
                self._pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token,
                )

            if self._pipeline is not None and self.device == "cuda":
                self._pipeline.to(torch.device("cuda"))
        except Exception as e:
            self._pipeline = None
            raise RuntimeError(
                f"Failed to load PyAnnote pipeline '{self.model_name}'. "
                f"Ensure valid HF_TOKEN is set. Error: {e}"
            )
        return self._pipeline

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """Run diarization and return sorted speaker segments."""
        pipeline = self._load_pipeline()
        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers

        diarization = pipeline(audio_path, **params)

        speaker_segments: List[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append(
                SpeakerSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
            )

        speaker_segments.sort(key=lambda s: s.start)
        return speaker_segments
