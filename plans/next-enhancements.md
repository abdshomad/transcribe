# Active Next Enhancements (100% Local Server Execution)

> **Current Milestone**: `v0.6.0` (`COMPLETED` ✅ — Archived to [`./roadmaps/v0.6.0-milestone.md`](./roadmaps/v0.6.0-milestone.md))  
> **Domain Documentation**:  
> - [`../docs/features/core/next-gen-asr-and-audio-llms.md`](../docs/features/core/next-gen-asr-and-audio-llms.md)  
> - [`../docs/features/ingestion/gdrive-folder-batch-and-combine.md`](../docs/features/ingestion/gdrive-folder-batch-and-combine.md)  
> - [`../docs/features/core/local-llm-intelligence.md`](../docs/features/core/local-llm-intelligence.md)

---

## Active Roadmap: Advanced Diarization & Realtime Intelligence (`v0.7.0`)

> **Execution Protocol**: Strict UI/UX First ➔ Polymorphic Engine Implementation ➔ Automated Playwright E2E Tests ➔ Record any defect/gap in [`../issues/`](../issues/) ➔ Fix & Iterate until 100% Success.

### Phase 1: Real-time Multi-Speaker Pyannote v3.1 Upgrade
- [ ] [DIAR-001] Upgrade local Pyannote embedding pipeline with clustering threshold calibration knobs.
- [ ] [DIAR-002] Implement speaker profile enrollment and persistent voiceprint naming (`history.db`).
- [ ] [DIAR-003] Add interactive Web UI speaker reassignment and color-coded voiceprint badges.

---

### Phase 2: Live Mic WebSocket Streaming & VAD Chunking
- [ ] [LIVE-001] Implement `/api/stream/ws` bidirectional WebSocket audio streaming endpoint.
- [ ] [LIVE-002] Add Silero-VAD real-time voice boundary chunking with sub-200ms latency.
- [ ] [LIVE-003] Implement browser WebRTC / AudioWorklet capture node in Web UI.

---

### Phase 3: Dedicated LLM MOM Pipeline Refinements
- [ ] [MOM-REF-001] Fine-tune local LLM prompt engineering for multi-speaker conflict resolution and action item assignment.
- [ ] [MOM-REF-002] Re-enable Web UI MOM button when `ENABLE_MOM=true` with live health-check indicator dot.

---

## Backlog & Excluded Items (Not Implemented / No Plan to Implement)

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
