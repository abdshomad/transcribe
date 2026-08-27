# Feature: Speech-to-Text & Speaker Diarization

- **ASR Engine**: Faster-Whisper (CTranslate2) with GPU CUDA acceleration & CPU fallback (`src/transcribe/transcriber.py`).
- **Diarization**: PyAnnote.Audio 3.1 neural diarizer with fallback clustering (`src/transcribe/diarizer.py`).
- **Alignment**: Temporal intersection mapper merging ASR word intervals and speaker turns (`src/transcribe/aligner.py`).
