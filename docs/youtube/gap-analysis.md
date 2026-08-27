# YouTube Playlist ASR & Voice Model Gap Analysis

**Playlist Source**: [AI Voice (ALL) 2026 Playlist](https://www.youtube.com/playlist?list=PLIsFyaUd-B31sk5X4_XvjvqApKUGvkTAA)

## 1. Discovered Speech Models & Technologies in Playlist

| Technology / Model | Family | Primary Modality | Status in Repo |
|:---|:---|:---|:---|
| **Whisper Standard (tiny -> large-v3)** | OpenAI / CTranslate2 | ASR / STT | Supported (`MODEL_CATALOG`) |
| **Whisper Turbo / large-v3-turbo** | OpenAI / CTranslate2 | ASR / STT | Supported (`MODEL_CATALOG`) |
| **Moonshine Voice (base/tiny)** | UsefulSensors | Edge ASR | Supported (`MODEL_CATALOG`) |
| **Voxtral Mini 3B / 4B Realtime** | Mistral AI | Edge Real-time ASR | Supported (`MODEL_CATALOG`) |
| **SenseVoice-Small** | Alibaba FunAudioLLM | Fast Multilingual ASR (50x RTF) | **New Addition** |
| **NVIDIA Nemotron Speech ASR** | NVIDIA NeMo | Cache-Aware Streaming RNN-T | **New Addition** |
| **Microsoft VibeVoice-ASR** | Microsoft | 60-min Long-form Acoustic ASR | **New Addition** |
| **MOSS-SATS ASR** | Fudan OpenMOSS | Target-Speaker Diarized ASR | **New Addition** |
| **ElevenLabs Scribe v2** | ElevenLabs API | Multilingual Diarized Cloud ASR | **New Addition** |
| **Amazon Transcribe** | AWS | Real-Time & Batch Cloud ASR | **New Addition** |
| **Qwen3-Audio / Qwen-TTS** | Alibaba Qwen | Audio Understanding & TTS | **New Addition** |
| **Tencent Covo-Audio 7B** | Tencent | End-to-end Speech Model | **New Addition** |
| **Whisper.cpp (GGML)** | Georgi Gerganov | Embedded / Edge ASR | Supported (`whisper.cpp`) |
| **Deepgram Nova-2** | Deepgram API | Cloud ASR | Supported (`MODEL_CATALOG`) |
| **Google Gemini Audio (Omni)** | Google Cloud API | Multimodal Audio Understanding | Supported (`MODEL_CATALOG`) |

---

## 2. Integrated Model Updates in `MODEL_CATALOG`

The following models from the playlist analysis are integrated into `src/transcribe/models.py`:
1. `sensevoice-small` (`FunAudioLLM/SenseVoiceSmall`): High-speed multilingual ASR with audio event detection.
2. `nvidia-nemotron-speech-asr` (`nvidia/nemotron-speech-asr`): Low-latency cache-aware ASR.
3. `microsoft-vibevoice-asr` (`microsoft/VibeVoice-ASR`): 60-minute long-form single pass acoustic ASR.
4. `moss-sats-diarized-asr` (`fnlp/MOSS-SATS`): Target-speaker speech recognition with integrated diarization.
5. `elevenlabs-scribe-v2`: ElevenLabs 99-language diarized cloud API.
6. `amazon-transcribe`: AWS Cloud speech-to-text integration.
7. `tencent-covo-audio-7b`: Tencent open end-to-end voice AI.
