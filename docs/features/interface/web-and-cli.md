# Feature: Web UI, REST API, SSE Streaming & CLI

- **Web Application**: Tailwind CSS single-page interface with live SSE timeline and virtualized window (`src/transcribe/web.py`).
- **REST & SSE API**: Endpoints for `/api/transcribe-stream`, `/api/sources`, `/api/compare`, and history CRUD (`src/transcribe/server.py`).
- **Exporters**: Real-time serializers for Markdown, Subtitles (SRT, VTT), Plain Text, and JSON.
- **CLI**: Rich/Typer terminal commands (`transcribe` and `audio-transcribe`).
