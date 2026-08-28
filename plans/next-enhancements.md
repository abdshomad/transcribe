# Active Next Enhancements (100% Local Server Execution)

> **Current Milestone**: `v0.5.0` (`COMPLETED` ✅ — Archived to [`./roadmaps/v0.5.0-milestone.md`](./roadmaps/v0.5.0-milestone.md))  
> **Domain Documentation**: [`../docs/features/interface/universal-model-selector.md`](../docs/features/interface/universal-model-selector.md)

---

## Active Roadmap: FireRed Team ASR & Audio-LLM Engine Integration (`v0.6.0`)

> **Execution Protocol**: Strict UI/UX First ➔ Polymorphic Engine Implementation ➔ Automated Playwright E2E Tests ➔ Record any defect/gap in [`../issues/`](../issues/) ➔ Fix & Iterate until 100% Success.

### Phase 1: Core Engine Architecture & Factory Registration
- [x] [RED-ENG-001] [DONE] Implement `FireRedTranscriber` in [`src/transcribe/engines/firered.py`](../src/transcribe/engines/firered.py) supporting `FireRedASR-AED-L` (1.1B Conformer-AED) and `FireRedASR-LLM-L` (7.8B Encoder-Adapter-LLM) with 16kHz resample and segment word interpolation.
- [x] [RED-ENG-002] [DONE] Register `FireRedTranscriber` in [`src/transcribe/engines/factory.py`](../src/transcribe/engines/factory.py) with alias mapping (`fireredasr-aed-l`, `fireredasr-llm-l`, `fireredaudio-9b`).
- [x] [RED-ENG-003] [DONE] Update Model Catalog in [`src/transcribe/models.py`](../src/transcribe/models.py) with `FireRed Team` family, VRAM estimates, parameter counts, and local cache checking.
- [x] [RED-ENG-004] [DONE] Add unit tests in `tests/test_engines_firered.py` and ensure cyclomatic complexity stays $\le 10$ across all routines.

---

### Phase 2: Web UI/UX Cascaded Selectors & Adaptive Knobs
- [x] [RED-UI-001] [DONE] Mount `"FireRed Team"` architecture in Web UI cascading selectors ([`src/transcribe/web.py`](../src/transcribe/web.py)) with real-time badges (Params, VRAM, Speed).
- [x] [RED-UI-002] [DONE] Implement FireRed adaptive settings panel (Beam Search Size for AED, Temperature & Max New Tokens for LLM variant).
- [x] [RED-UI-003] [DONE] Wire FireRed dynamic form parameters into `/api/transcribe-stream` and [`src/transcribe/pipeline.py`](../src/transcribe/pipeline.py).
- [x] [RED-UI-004] [DONE] Update Timeline Quick-Switch pills to enable 1-click comparison between Whisper Turbo, SenseVoice, and FireRedASR.

---

### Phase 3: Meta OmniASR (1,600+ Langs) & MMS Multi-Variant Scaling
- [x] [META-OMNI-001] [DONE] Integrate Meta MMS variants (`meta-mms-1b-all`, `meta-mms-1b-fl102`, `meta-mms-300m`) and Meta OmniASR (`omniasr-ctc-300m`, `omniasr-ctc-1b`) in [`src/transcribe/engines/transformers_ctc.py`](../src/transcribe/engines/transformers_ctc.py).
- [x] [META-OMNI-002] [DONE] Add remapped Wav2Vec2/Lasr CTC loader and SentencePiece decoding for OmniASR.
- [x] [META-OMNI-003] [DONE] Add `"Meta OmniASR"` and `"Meta MMS"` cascading dropdown options with real-time badges in Web UI ([`src/transcribe/web.py`](../src/transcribe/web.py)).
- [x] [META-OMNI-004] [DONE] Add unit tests in `tests/test_engines_transformers_ctc.py` (6/6 passing).

---

### Phase 4: VoiceMem Dual-Brain Voice Memory & Emotion Architecture (`voicemem`)
- [x] [VMEM-001] [DONE] Implement `VoiceMemTranscriber` in [`src/transcribe/engines/voicemem_engine.py`](../src/transcribe/engines/voicemem_engine.py) wrapping `voicemem` audio perception (ASR + SER Emotion + Speaker Identification + Scene Detection).
- [x] [VMEM-002] [DONE] Register `voicemem-normal` and `voicemem-realtime` in [`src/transcribe/engines/factory.py`](../src/transcribe/engines/factory.py) and [`src/transcribe/models.py`](../src/transcribe/models.py).
- [x] [VMEM-003] [DONE] Mount `"Tsinghua VoiceMem"` in Web UI with Emotion tags (`😊 Happy`, `😠 Angry`, `😔 Sad`, `😐 Neutral`) and Scene tags (`🏢 Office`, `🚗 Vehicle`, `🏠 Indoor`).
- [x] [VMEM-004] [DONE] Write unit tests in `tests/test_engines_voicemem.py` with Cyclomatic Complexity $\le 10$.

### Phase 5: Whisper.cpp GGML / GGUF High-Efficiency C++ Engine
- [x] [WCPP-001] [DONE] Implement `WhisperCppTranscriber` in [`src/transcribe/engines/whisper_cpp.py`](../src/transcribe/engines/whisper_cpp.py) supporting GGML/GGUF quantized models (`whispercpp-tiny`, `whispercpp-base`, `whispercpp-small`, `whispercpp-medium`, `whispercpp-turbo`, `whispercpp-large-v3`).
- [x] [WCPP-002] [DONE] Register `whispercpp-*` aliases in [`src/transcribe/engines/factory.py`](../src/transcribe/engines/factory.py) and [`src/transcribe/models.py`](../src/transcribe/models.py).
- [x] [WCPP-003] [DONE] Mount `"Whisper.cpp (GGML)"` family in Web UI cascading selectors ([`src/transcribe/web.py`](../src/transcribe/web.py)) with real-time badges (GGML Quant, Threads, CPU/Metal optimization).
- [x] [WCPP-004] [DONE] Write unit tests in `tests/test_engines_whisper_cpp.py` with Cyclomatic Complexity $\le 10$.

---

### Phase 6: NVIDIA NeMo & Sherpa-ONNX Engine Family (Parakeet-TDT & Nemotron Speech)
- [x] [NEMO-001] [DONE] Implement `SherpaNemoTranscriber` in [`src/transcribe/engines/nemo_engine.py`](../src/transcribe/engines/nemo_engine.py) supporting NVIDIA Parakeet (`parakeet-tdt-0.6b`, `parakeet-tdt-1.1b`, `parakeet-ctc-0.6b`, `parakeet-ctc-1.1b`) and NVIDIA Nemotron (`nemotron-speech-3.5`) via `sherpa-onnx` runtime.
- [x] [NEMO-002] [DONE] Register `nvidia-parakeet-*` and `nvidia-nemotron-*` in [`src/transcribe/engines/factory.py`](../src/transcribe/engines/factory.py) and [`src/transcribe/models.py`](../src/transcribe/models.py).
- [x] [NEMO-003] [DONE] Mount `"NVIDIA NeMo / Parakeet"` family in Web UI cascading dropdowns ([`src/transcribe/web.py`](../src/transcribe/web.py)) with real-time badges (TDT RNN-T, 1.1B params, ~25x RTF).
- [x] [NEMO-004] [DONE] Write unit tests in `tests/test_engines_nemo.py` with Cyclomatic Complexity $\le 10$.

---

### Phase 7: NVIDIA Nemotron-Labs-Audex-2B Audio-LLM Integration (`nvidia/Nemotron-Labs-Audex-2B`)
- [x] [AUDEX-001] [DONE] Implement `AudexTranscriber` in [`src/transcribe/engines/audex_engine.py`](../src/transcribe/engines/audex_engine.py) supporting `nvidia/Nemotron-Labs-Audex-2B` (2B Compact Unified Audio-Text LLM with 128k context).
- [x] [AUDEX-002] [DONE] Support dual execution modes: **Instruct Mode** (verbatim ASR) and **Thinking Mode** (`<think>...</think>` speech reasoning).
- [x] [AUDEX-003] [DONE] Register `nemotron-audex-2b` in [`src/transcribe/engines/factory.py`](../src/transcribe/engines/factory.py) and [`src/transcribe/models.py`](../src/transcribe/models.py).
- [x] [AUDEX-004] [DONE] Mount Audex-2B in Web UI ([`src/transcribe/web.py`](../src/transcribe/web.py)) with Thinking Mode toggle pill and temperature knobs.
- [x] [AUDEX-005] [DONE] Add unit tests in `tests/test_engines_audex.py` with Cyclomatic Complexity $\le 10$.

---

### Phase 8: Issue Logging & Defect Tracking Protocol
- [ ] [QA-001] Always record any test failure, browser glitch, or model download exception in [`../issues/`](../issues/) using format `issues/{num}-{slug}.md`.
- [ ] [QA-002] Iterate on code changes and re-test until all issue reproduction steps succeed with 0 errors.
- [ ] [QA-003] Mark resolved issues with resolution summary and update feature docs in [`../docs/features/`](../docs/features/).

---

### Phase 9: Google Drive Folder Batch & Auto-Resume Processing
- [x] [BATCH-001] [DONE] Implement `extract_gdrive_folder_id` and folder content discovery with media candidate filtering (`.m4a`, `.mp3`, `.wav`, `.mp4`, etc., OR keyword matching for Google Meet recordings without extensions) in [`src/transcribe/downloader.py`](../src/transcribe/downloader.py).
- [x] [BATCH-002] [DONE] Implement HTTP `Range: bytes={start}-` auto-resume for interrupted downloads in [`src/transcribe/downloader.py`](../src/transcribe/downloader.py).
- [x] [BATCH-003] [DONE] Implement automatic checkpoint auto-resuming from `history.db` (`last_processed_time > 0`) in [`src/transcribe/pipeline.py`](../src/transcribe/pipeline.py) and [`src/transcribe/cli.py`](../src/transcribe/cli.py).
- [x] [BATCH-004] [DONE] Implement CLI interactive batch confirmation (`[Y/n]` prompt and `--yes` / `-y` flag) with per-file fault isolation and summary table in [`src/transcribe/cli.py`](../src/transcribe/cli.py).
- [x] [BATCH-005] [DONE] Implement Web UI batch queue support in [`src/transcribe/server.py`](../src/transcribe/server.py).
- [x] [BATCH-006] [DONE] Add unit tests in `tests/test_downloader.py`, `tests/test_history.py`, and `tests/test_cyclomatic_complexity.py` with Cyclomatic Complexity $\le 10$.

### Phase 10: Live Media Conversion Progress Tracking
- [x] [CONV-001] [DONE] Implement `probe_media_duration` and `convert_slice_to_wav_16k_with_progress` in [`src/transcribe/audio.py`](../src/transcribe/audio.py) and [`src/transcribe/pipeline.py`](../src/transcribe/pipeline.py) with FFmpeg `-progress pipe:1` stdout streaming.
- [x] [CONV-002] [DONE] Wire conversion progress callback into `_prepare_pipeline_audio` and `pipeline.process()` emitting `stage: "converting"` and compact percent (`⚙️ Converting media... X%`).
- [x] [CONV-003] [DONE] Implement live conversion progress display in CLI ([`src/transcribe/cli.py`](../src/transcribe/cli.py)) using Rich Progress bar.
- [x] [CONV-004] [DONE] Support conversion progress events in Web UI SSE streaming in [`src/transcribe/server.py`](../src/transcribe/server.py).
- [x] [CONV-005] [DONE] Add unit tests in `tests/test_audio_conversion_progress.py` with Cyclomatic Complexity $\le 10$.

---

## Backlog & Excluded Items (Not Implemented / No Plan to Implement)

### Excluded Post-Processing & Rewrite Layers
> *Decision: Raw acoustic transcription fidelity and precise word alignment are prioritized. Text post-rewrite LLMs are excluded to preserve original verbatim audio integrity.*
- [ ] [LLM-REF-001] [NOT IMPLEMENTED / NO PLAN TO IMPLEMENT] Local AI Post-Processing & Transcript Refinement Layer (Fluid-1 / Ollama / vLLM transcript rewrite)
- [ ] [LLM-REF-002] [NOT IMPLEMENTED / NO PLAN TO IMPLEMENT] Automatic casing, punctuation, and disfluency alteration LLM engine
- [ ] [LLM-REF-003] [NOT IMPLEMENTED / NO PLAN TO IMPLEMENT] AI Auto-Formatting & Polishing toggle in Web UI

### Cloud-Hosted STT APIs (Deferred / Excluded)
> *Note: Cloud integrations are deferred to keep the system 100% self-hosted on local server GPUs.*
- [ ] [CLOUD-001] [NOT IMPLEMENTED] OpenAI Whisper Cloud API (`whisper-1`, `gpt-4o-transcribe`)
- [ ] [CLOUD-002] [NOT IMPLEMENTED] Google Gemini 2.5 Flash Audio Understanding API
- [ ] [CLOUD-003] [NOT IMPLEMENTED] Deepgram Nova-2 REST / Streaming STT API
- [ ] [CLOUD-004] [NOT IMPLEMENTED] ElevenLabs Scribe v2 Diarized Transcription API
- [ ] [CLOUD-005] [NOT IMPLEMENTED] AWS Amazon Transcribe Asynchronous Batch Job API
- [ ] [CLOUD-006] [NOT IMPLEMENTED] Microsoft Azure Cognitive Services Speech-to-Text API
- [ ] [CLOUD-007] [NOT IMPLEMENTED] Groq Cloud LPU Accelerated Whisper API
- [ ] [CLOUD-008] [NOT IMPLEMENTED] AssemblyAI Universal-2 / Conformer-2 Speech API
- [ ] [CLOUD-009] [NOT IMPLEMENTED] Rev.ai Enterprise Speech Recognition API
