# Issue 004: PyAnnote 3.1 Gated Repository Access & DiarizeOutput Track Resolution

## Summary
When running `PyAnnoteDiarizer` on GPU with `pyannote/speaker-diarization-3.1`, two issues were encountered:
1. **Gated Repo 403 Forbidden**: In addition to `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, PyAnnote downloads PLDA weights from `pyannote/speaker-diarization-community-1`, which requires explicit user terms acceptance on Hugging Face.
2. **`DiarizeOutput` Attribute Error**: In `pyannote.audio` 3.1+, the pipeline returns a `DiarizeOutput` dataclass rather than a direct `Annotation`, causing `diarization.itertracks(yield_label=True)` to raise `AttributeError: 'DiarizeOutput' object has no attribute 'itertracks'`.

## Root Cause
- Hugging Face gating requires accepting terms on all 3 sub-repositories for the pipeline to download community PLDA weights.
- `src/transcribe/diarizer.py` called `.itertracks()` directly on the pipeline return value without unpacking `.speaker_diarization`.

## Resolution
1. **Secret & Environment Export**: Updated `_read_hf_token()` in [`src/transcribe/diarizer.py`](../src/transcribe/diarizer.py) to read `HF_TOKEN` from `.secrets` and set `os.environ["HF_TOKEN"]` and `os.environ["HUGGING_FACE_HUB_TOKEN"]` globally.
2. **DiarizeOutput Unpacking**: Updated `PyAnnoteDiarizer.diarize()` to extract `annotation = getattr(diarization, "speaker_diarization", diarization)` before iterating over tracks.
3. **GPU Device Handling**: Passed `torch.device("cuda")` into `pipeline.to(...)`.

## Verification
- Verified end-to-end speaker diarization on `data/samples/english_jfk_16k.wav`, accurately identifying speaker turns on CUDA GPU with 0 errors.
