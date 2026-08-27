# Issue 002: SenseVoice `use_itn` Duplicate Keyword Argument in `AutoModel.generate()`

## Problem Summary
When transcribing with Alibaba SenseVoice (`sensevoice-small`), the execution failed with:
`TypeError: funasr.auto.auto_model.AutoModel.generate() got multiple values for keyword argument 'use_itn'`

## Root Cause
Both `self.use_itn` (explicit argument) and `kwargs['use_itn']` (forwarded from pipeline/web dynamic parameter) were passed into `model.generate(use_itn=use_itn, **kwargs)`.

## Resolution
1. Update `_run_sensevoice_model()` in `src/transcribe/engines/sensevoice.py` to pop `use_itn` from `call_kwargs`.
2. Re-run SenseVoice E2E browser screenshot test.

## Status
RESOLVED ✅
