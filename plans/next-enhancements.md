# Active Next Enhancements

> **Technical Architecture Reference**: [`docs/features/core/multi-engine-transcriber-architecture.md`](file:///home/aiserver/LABS/AI-VOICE/audio-to-transcription/docs/features/core/multi-engine-transcriber-architecture.md)

---

## Phase 1: Modular Engine Framework & Transcriber Factory
- [ ] [ASR-005] Base Transcriber Interface: Define abstract `BaseTranscriber` in `src/transcribe/engines/base.py` with standard `transcribe(audio_path, language, ...)` returning `Tuple[List[TranscriptSegment], str, float]`.
- [ ] [ASR-006] Engine Factory & Registry: Create dynamic `EngineRegistry` in `src/transcribe/engines/factory.py` with lazy dynamic imports for zero-overhead initialization.
- [ ] [ASR-007] Faster-Whisper Engine Refactor: Wrap `FasterWhisperTranscriber` as `FasterWhisperEngine(BaseTranscriber)` in `src/transcribe/engines/faster_whisper.py`.

## Phase 2: Indonesian Wav2Vec2 & Meta MMS Acoustic CTC Engines
- [ ] [ASR-008] Transformers CTC Base Engine: Implement `TransformersCTCEngine` in `src/transcribe/engines/transformers_ctc.py` supporting PyTorch CTC models with chunked windowing.
- [ ] [ASR-009] Regional Indonesian Acoustic Models: Integrate `indonesian-wav2vec2-regional` (`indonesian-nlp/wav2vec2-indonesian-javanese-sundanese`) and `indonesian-wav2vec2-large-xlsr`.
- [ ] [ASR-010] Meta MMS Omnilingual ASR: Integrate `meta-omnilingual-asr` (`facebook/mms-1b-all`) with language-specific adapter vocabulary switching for 100+ languages.

## Phase 3: Alibaba FunAudioLLM SenseVoice Engine
- [ ] [ASR-011] SenseVoice-Small Engine: Implement `SenseVoiceEngine` in `src/transcribe/engines/sensevoice.py` supporting `FunAudioLLM/SenseVoiceSmall` (50x real-time factor).
- [ ] [ASR-012] Emotion (SER) & Event (AED) Parsing: Implement structured markup extractor for SenseVoice tags (`<|HAPPY|>`, `<|LAUGHTER|>`, `<|APPLAUSE|>`, `<|CRY|>`, `<|MUSIC|>`).

## Phase 4: UsefulSensors Moonshine Edge ONNX Engine
- [ ] [ASR-013] Moonshine ONNX Engine: Implement `MoonshineEngine` in `src/transcribe/engines/moonshine.py` utilizing ONNX Runtime for zero-dependency edge CPU/GPU execution.
- [ ] [ASR-014] Moonshine Model Variants: Support `moonshine-tiny` and `moonshine-base` variable-length acoustic encoding.

## Phase 5: Cloud-Hosted STT API Provider Integrations
- [ ] [ASR-015] Cloud STT Base Provider: Create async/sync HTTP client interface in `src/transcribe/engines/cloud/base.py` with environment key configuration.
- [ ] [ASR-016] OpenAI Whisper API: Implement `OpenAIWhisperEngine` supporting `whisper-1` and `gpt-4o-transcribe` with timestamp granularities.
- [ ] [ASR-017] Google Gemini Audio API: Implement `GoogleGeminiAudioEngine` using Google GenAI SDK for multimodal audio understanding and transcription.
- [ ] [ASR-018] Deepgram Nova-2 API: Implement `DeepgramEngine` for ultra-fast streaming and batch Nova-2 ASR.
- [ ] [ASR-019] ElevenLabs Scribe v2 API: Implement `ElevenLabsScribeEngine` supporting 99-language diarized cloud transcription.
- [ ] [ASR-020] AWS Amazon Transcribe: Implement `AmazonTranscribeEngine` using `boto3` for S3 upload and asynchronous transcription jobs.

## Phase 6: NVIDIA NeMo FastConformer & Cache-Aware Nemotron
- [ ] [ASR-021] NVIDIA NeMo Parakeet TDT Engine: Implement `NeMoEngine` in `src/transcribe/engines/nemo.py` supporting `nvidia/parakeet-tdt-0.6b-v3` FastConformer-TDT architecture.
- [ ] [ASR-022] NVIDIA Nemotron Streaming Engine: Implement low-latency streaming RNN-T inference with cache-aware chunk processing.

## Phase 7: Long-Form & Next-Gen Audio-LLMs
- [ ] [ASR-023] Microsoft VibeVoice-ASR Engine: Implement `VibeVoiceEngine` in `src/transcribe/engines/vibevoice.py` for unified 60-minute single-pass acoustic ASR.
- [ ] [ASR-024] Kyutai Moshi / STT Engine: Implement `KyutaiEngine` in `src/transcribe/engines/kyutai.py` supporting duplex speech recognition.
- [ ] [ASR-025] Mistral Voxtral Mini 3B Engine: Implement `VoxtralEngine` in `src/transcribe/engines/voxtral.py` for edge speech understanding.
- [ ] [ASR-026] Fudan MOSS-SATS Engine: Implement `MossSatsEngine` in `src/transcribe/engines/moss_sats.py` for target-speaker diarized ASR.
- [ ] [ASR-027] Alibaba Qwen3-Audio & Tencent Covo-Audio: Implement multimodal Audio-LLM conversational speech extractors.

## Phase 8: Unified Benchmark Matrix & Web UI Integration
- [ ] [ASR-028] Universal Benchmark Runner Extension: Update `scripts/benchmark_all_models.py` to evaluate non-Whisper backends (CTC, SenseVoice, Moonshine, Cloud APIs) side-by-side with WER/CER and RTF metrics.
- [ ] [ASR-029] Web UI Model Selector & Capability Badges: Update Web UI with multi-family dropdowns, capability badges (`[Cloud]`, `[Edge]`, `[SER]`, `[CTC]`), and multi-engine live execution.
- [ ] [ASR-030] Full End-to-End Test Suite: Implement unit and mock tests in `tests/test_engines/` for all 8 engine categories.
