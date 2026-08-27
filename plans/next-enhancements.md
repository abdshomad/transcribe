# Active Next Enhancements (100% Local Server Execution)

> **Current Milestone**: `v0.4.0` (`COMPLETED` ✅ — Archived to [`./roadmaps/v0.4.0-milestone.md`](./roadmaps/v0.4.0-milestone.md))  
> **Domain Documentation**: [`../docs/features/core/polymorphic-engines.md`](../docs/features/core/polymorphic-engines.md)

---

## Active Roadmap: Universal Web Model & Variant Hierarchy Selector (`v0.5.0`)

### Core Architecture & Streaming Pipeline
- [ ] [WEB-001] Extend `/api/models` endpoint with rich metadata: family, variant sizes, quantization options (`float16`, `int8`, `int8_float16`), VRAM estimates, parameter count, and local server cache status (`is_cached: bool`).
- [ ] [WEB-002] Implement Dynamic Cascaded Selectors in Web UI ([`src/transcribe/web.py`](../src/transcribe/web.py)): Family/Architecture -> Model Variant / Size -> Quantization / Compute Type with real-time parameter & VRAM badges.
- [ ] [WEB-003] Implement Dynamic Adaptive Settings Panel in Web UI that contextually mounts family-specific knobs (Whisper: Beam Size / VAD; SenseVoice: Emotion / ITN; Meta MMS: Adapter Lang; CTC: Chunk Window).
- [ ] [WEB-004] Implement Automatic On-Demand Server Model Download with real-time SSE progress bar feedback in Web UI during transcription initialization if model is not yet cached.
- [ ] [WEB-005] Update Timeline Quick-Switch Pills to dynamically generate comparative 1-click re-transcribe buttons for all sizes/variants of the selected family and top cross-family alternatives.
- [ ] [WEB-006] Wire dynamic form parameters (`compute_type`, `beam_size`, `vad_filter`, `use_itn`, `chunk_length_s`) from Web UI into `/api/transcribe-stream` and [`src/transcribe/pipeline.py`](../src/transcribe/pipeline.py).
- [ ] [WEB-007] Add end-to-end unit and integration tests in `tests/test_server_models_selection.py` and ensure cyclomatic complexity stays $\le 10$ across all new and modified routines.

---

### E2E Browser Visual Verification Suite (`screenshots/e2e/`)
- [ ] [E2E-001] Create automated Playwright E2E runner [`tests/e2e/test_web_models_screenshots.py`](../tests/e2e/test_web_models_screenshots.py) with structured screenshot output path: `screenshots/e2e/{model}/{model}-{size}-{variant}/{stepnum}-{short-step-name}.jpg`.
- [ ] [E2E-002] Implement 5 standard lifecycle screenshot steps per model variant:
  - `01-initial-state.jpg`: Initial clean UI load.
  - `02-model-configured.jpg`: Cascaded model, size variant, and compute type selected with active badges & adaptive knobs.
  - `03-file-uploaded.jpg`: Audio sample loaded and ready to transcribe.
  - `04-streaming-progress.jpg`: Active SSE streaming with real-time progress bar, live badges, and partial segments.
  - `05-completed-results.jpg`: Completed transcript timeline, speaker badges, and export action bar.
- [ ] [E2E-003] Verify all generated JPG screenshots in `screenshots/e2e/` and create visual walkthrough report with carousels.

---

### Sub-Plan 1: Faster-Whisper Family (OpenAI / CTranslate2)
- [ ] [MOD-FW-01] Whisper Tiny (`tiny` • 39M • float16 / int8 / int8_float16) + E2E screenshots in `screenshots/e2e/whisper/whisper-tiny-default/`
- [ ] [MOD-FW-02] Whisper Base (`base` • 74M • float16 / int8 / int8_float16) + E2E screenshots in `screenshots/e2e/whisper/whisper-base-default/`
- [ ] [MOD-FW-03] Whisper Small (`small` • 244M • float16 / int8 / int8_float16) + E2E screenshots in `screenshots/e2e/whisper/whisper-small-default/`
- [ ] [MOD-FW-04] Whisper Medium (`medium` • 769M • float16 / int8 / int8_float16) + E2E screenshots in `screenshots/e2e/whisper/whisper-medium-default/`
- [ ] [MOD-FW-05] Whisper Turbo (`turbo` • 809M • float16 / int8 / int8_float16) + E2E screenshots in `screenshots/e2e/whisper/whisper-turbo-default/`
- [ ] [MOD-FW-06] Whisper Large-v3 (`large-v3` • 1550M • float16 / int8 / int8_float16) + E2E screenshots in `screenshots/e2e/whisper/whisper-large-v3-default/`
- [ ] [MOD-FW-07] Distil-Whisper Small English (`distil-small.en` • 166M • English Fast) + E2E screenshots in `screenshots/e2e/whisper/distil-whisper-small-en/`
- [ ] [MOD-FW-08] Distil-Whisper Medium English (`distil-medium.en` • 394M • English Fast) + E2E screenshots in `screenshots/e2e/whisper/distil-whisper-medium-en/`

---

### Sub-Plan 2: Alibaba SenseVoice Family (Rich Speech Recognition)
- [ ] [MOD-SV-01] SenseVoice-Small FP16 (`sensevoice-small` • SER Emotion & AED Events) + E2E screenshots in `screenshots/e2e/sensevoice/sensevoice-small-fp16/`
- [ ] [MOD-SV-02] SenseVoice-Small INT8 (`sensevoice-small` • Quantized Fast) + E2E screenshots in `screenshots/e2e/sensevoice/sensevoice-small-int8/`

---

### Sub-Plan 3: UsefulSensors Moonshine Family (Edge ASR)
- [ ] [MOD-MS-01] Moonshine Tiny ONNX (`moonshine-tiny` • 27M • Zero-Overhead ONNX) + E2E screenshots in `screenshots/e2e/moonshine/moonshine-tiny-onnx/`
- [ ] [MOD-MS-02] Moonshine Base ONNX (`moonshine-base` • 61M • Accurate ONNX) + E2E screenshots in `screenshots/e2e/moonshine/moonshine-base-onnx/`
- [ ] [MOD-MS-03] Moonshine Tiny PyTorch (`moonshine-tiny` • 27M • Transformers Fallback) + E2E screenshots in `screenshots/e2e/moonshine/moonshine-tiny-torch/`
- [ ] [MOD-MS-04] Moonshine Base PyTorch (`moonshine-base` • 61M • Transformers Fallback) + E2E screenshots in `screenshots/e2e/moonshine/moonshine-base-torch/`

---

### Sub-Plan 4: Meta MMS Omnilingual Family (Multilingual CTC)
- [ ] [MOD-MMS-01] Meta MMS-1B Omnilingual (`meta-omnilingual-asr` • 1B • 100+ Langs) + E2E screenshots in `screenshots/e2e/mms/mms-1b-all/`

---

### Sub-Plan 5: Indonesian Regional & Dialects CTC Family
- [ ] [MOD-CTC-01] Indonesian Wav2Vec2 Regional (`indonesian-wav2vec2-regional` • ID/JV/SU Native) + E2E screenshots in `screenshots/e2e/wav2vec2/wav2vec2-regional-id-jv-su/`
- [ ] [MOD-CTC-02] Indonesian Wav2Vec2 Large XLSR (`indonesian-wav2vec2-large-xlsr` • 53-Lang Indonesian) + E2E screenshots in `screenshots/e2e/wav2vec2/wav2vec2-large-xlsr-id/`

---

## Backlog: Cloud-Hosted STT APIs (Deferred / Not Implemented)
> *Note: These cloud integrations are deferred to keep the system 100% self-hosted on local server GPUs.*
- [ ] [CLOUD-001] [NOT IMPLEMENTED] OpenAI Whisper Cloud API (`whisper-1`, `gpt-4o-transcribe`)
- [ ] [CLOUD-002] [NOT IMPLEMENTED] Google Gemini 2.5 Flash Audio Understanding API
- [ ] [CLOUD-003] [NOT IMPLEMENTED] Deepgram Nova-2 REST / Streaming STT API
- [ ] [CLOUD-004] [NOT IMPLEMENTED] ElevenLabs Scribe v2 Diarized Transcription API
- [ ] [CLOUD-005] [NOT IMPLEMENTED] AWS Amazon Transcribe Asynchronous Batch Job API
- [ ] [CLOUD-006] [NOT IMPLEMENTED] Microsoft Azure Cognitive Services Speech-to-Text API
- [ ] [CLOUD-007] [NOT IMPLEMENTED] Groq Cloud LPU Accelerated Whisper API
- [ ] [CLOUD-008] [NOT IMPLEMENTED] AssemblyAI Universal-2 / Conformer-2 Speech API
- [ ] [CLOUD-009] [NOT IMPLEMENTED] Rev.ai Enterprise Speech Recognition API
