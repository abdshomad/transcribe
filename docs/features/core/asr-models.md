# Feature: ASR / STT Model Registry & Benchmark Suite

## 1. Multi-Family Model Catalog (`src/transcribe/models.py`)
The system maintains an exhaustive metadata catalog of **33 ASR architectures** spanning 5 size tiers:
- **Standard Whisper**: `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo`, `turbo`.
- **English-Only Models**: `tiny.en`, `base.en`, `small.en`, `medium.en`.
- **Distil-Whisper**: `distil-small.en`, `distil-medium.en`, `distil-large-v2`, `distil-large-v3` (up to 73.9x RTF).
- **Indonesian Fine-Tunes (CT2 Converted)**: `cahya-whisper-tiny-id`, `cahya-whisper-base-id`, `cahya-whisper-small-id`, `cahya-whisper-medium-id`, `cahya-faster-whisper-medium-id`, `indonesian-wav2vec2-regional`, `indonesian-wav2vec2-large-xlsr`.
- **Edge & Next-Gen Open ASR**: `moonshine-base`, `nvidia-parakeet-tdt-v3`, `kyutai-stt`, `meta-omnilingual-asr`, `voxtral-mini-3b`, `gemma-3n-audio`, `sensevoice-small`, `nvidia-nemotron-speech-asr`, `microsoft-vibevoice-asr`, `moss-sats-diarized-asr`, `qwen3-audio-stt`, `tencent-covo-audio-7b`.
- **Cloud Hosted APIs**: OpenAI Whisper API, Google Gemini Audio, Deepgram Nova-2, ElevenLabs Scribe v2, Amazon Transcribe.

## 2. CLI Model Management & Benchmarking
- `uv run transcribe models`: Displays interactive Rich table detailing architecture families, parameter counts, VRAM allocations, and language capabilities.
- `uv run python scripts/benchmark_all_models.py [--force] [--compute-types float16 int8_float16 int8]`: Exhaustive automated test harness.

## 3. Automated Benchmark Results (114/114 Runs, 100% Success)
- **Hardware**: NVIDIA L40 (CUDA)
- **Test Datasets**:
  - `proklamasi.wav` (Indonesian speech, 48.52s): `cahya-whisper-small-id` achieved **44.4% WER** (ranking #1 overall in Indonesian accuracy, outperforming OpenAI `large-v3` at 47.2%). Speeds reached **135.5x RTF** (`cahya-whisper-tiny-id` float16).
  - `jfk.wav` (English speech, 11.00s): Speeds up to **73.9x RTF** (`distil-small.en` int8_float16) and **9.5% WER** across Whisper tiers.
- **Detailed Reports**:
  - [`./all-models-benchmark-proklamasi.md`](./all-models-benchmark-proklamasi.md)
  - [`./all-models-benchmark-jfk.md`](./all-models-benchmark-jfk.md)

