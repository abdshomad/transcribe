# Issue 001: Hugging Face PyTorch Checkpoints Require CTranslate2 Conversion for Faster-Whisper

**Status**: `RESOLVED` ✅

## Description
When running `cahya/whisper-small-id` or `cahya/whisper-medium-id` with `FasterWhisperTranscriber`, the engine raised:
```text
Unable to open file 'model.bin' in model '/home/aiserver/.cache/huggingface/hub/models--cahya--whisper-small-id/snapshots/...'
```

## Root Cause
`faster-whisper` relies on the **CTranslate2** inference engine, which requires weights in binary CTranslate2 format (`model.bin` + `vocabulary.json`).
Standard Hugging Face models like `cahya/whisper-*-id` store weights in PyTorch (`model.safetensors` / `pytorch_model.bin`) rather than CTranslate2 format.

## Resolution Applied
1. Installed `transformers` into project environment.
2. Converted checkpoints to CTranslate2 `float16` weights:
   ```bash
   uv run ct2-transformers-converter --model cahya/whisper-small-id --output_dir data/models/cahya-whisper-small-id-ct2 --quantization float16 --force
   uv run ct2-transformers-converter --model cahya/whisper-medium-id --output_dir data/models/cahya-whisper-medium-id-ct2 --quantization float16 --force
   ```
3. Updated `src/transcribe/transcriber.py` to auto-resolve `MODEL_ALIASES` to `data/models/cahya-whisper-*-id-ct2`.
4. Added default language fallback `language="id"` for monolingual Indonesian fine-tuned models to prevent index error during multi-language sampling.
5. Benchmark result on `proklamasi.wav`: `cahya-whisper-small-id` achieved **44.4% WER** (ranking #1 overall in Indonesian speech accuracy, outperforming OpenAI `large-v3` at 47.2%).
