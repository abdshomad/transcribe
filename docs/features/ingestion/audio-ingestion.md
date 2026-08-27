# Feature: Audio Ingestion & Normalization

- **Formats**: WAV, MP3, MP4, FLAC, OGG, WEBM via FFmpeg and SoundFile (`src/transcribe/audio.py`).
- **Remote Ingestion**: Anonymous Google Drive (`gdown`) & YouTube/HTTP (`yt-dlp`) with caching in `data/downloads/`.
- **Upload Persistence**: Uploaded audio preserved in `data/uploads/` for zero-reupload multi-model re-transcription.
