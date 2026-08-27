# Comprehensive Benchmark Report: `jfk.wav`

> **Hardware**: NVIDIA L40  
> **Audio Sample**: `jfk.wav` (Duration: 11.00s)  
> **Date**: 2026-08-27 20:47:32  
> **Engine**: Faster-Whisper (CTranslate2) across 3 Quantization Modes

> **Ground Truth Reference**: "And so, my fellow Americans: ask not what your country can do for you—ask what you can do for your country."

---

## 1. Summary Comparison Matrix

| Rank | Model | Size Tier | Quantization | Family | Params | WER (%) | CER (%) | Time (s) | Speed (x RTF) | Lang (Prob) | Words |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **`tiny`** | **Tiny** | `int8_float16` | Whisper Standard | 39M | **9.52%** | 0.0% | 0.159s | **69.2x** | `en` (0.973) | 22 |
| 2 | **`tiny`** | **Tiny** | `int8` | Whisper Standard | 39M | **9.52%** | 0.0% | 0.161s | **68.5x** | `en` (0.973) | 22 |
| 3 | **`tiny`** | **Tiny** | `float16` | Whisper Standard | 39M | **9.52%** | 0.0% | 0.161s | **68.3x** | `en` (0.972) | 22 |
| 4 | **`distil-medium.en`** | **Medium** | `int8` | Distil-Whisper | 394M | **9.52%** | 0.0% | 0.168s | **65.6x** | `en` (1) | 22 |
| 5 | **`base.en`** | **Base** | `float16` | Whisper English | 74M | **9.52%** | 0.0% | 0.171s | **64.3x** | `en` (1) | 22 |
| 6 | **`tiny.en`** | **Tiny** | `int8` | Whisper English | 39M | **9.52%** | 0.0% | 0.172s | **63.8x** | `en` (1) | 22 |
| 7 | **`base.en`** | **Base** | `int8_float16` | Whisper English | 74M | **9.52%** | 0.0% | 0.176s | **62.4x** | `en` (1) | 22 |
| 8 | **`tiny.en`** | **Tiny** | `int8_float16` | Whisper English | 39M | **9.52%** | 0.0% | 0.179s | **61.4x** | `en` (1) | 22 |
| 9 | **`base`** | **Base** | `int8_float16` | Whisper Standard | 74M | **9.52%** | 0.0% | 0.187s | **58.8x** | `en` (0.928) | 22 |
| 10 | **`distil-medium.en`** | **Medium** | `float16` | Distil-Whisper | 394M | **9.52%** | 0.0% | 0.19s | **57.9x** | `en` (1) | 22 |
| 11 | **`distil-medium.en`** | **Medium** | `int8_float16` | Distil-Whisper | 394M | **9.52%** | 0.0% | 0.194s | **56.8x** | `en` (1) | 22 |
| 12 | **`tiny.en`** | **Tiny** | `float16` | Whisper English | 39M | **9.52%** | 0.0% | 0.194s | **56.7x** | `en` (1) | 22 |
| 13 | **`base`** | **Base** | `int8` | Whisper Standard | 74M | **9.52%** | 0.0% | 0.206s | **53.3x** | `en` (0.928) | 22 |
| 14 | **`small.en`** | **Small** | `int8` | Whisper English | 244M | **9.52%** | 0.0% | 0.207s | **53.0x** | `en` (1) | 22 |
| 15 | **`base.en`** | **Base** | `int8` | Whisper English | 74M | **9.52%** | 0.0% | 0.21s | **52.5x** | `en` (1) | 22 |
| 16 | **`small.en`** | **Small** | `float16` | Whisper English | 244M | **9.52%** | 0.0% | 0.214s | **51.3x** | `en` (1) | 22 |
| 17 | **`small.en`** | **Small** | `int8_float16` | Whisper English | 244M | **9.52%** | 0.0% | 0.216s | **51.0x** | `en` (1) | 22 |
| 18 | **`distil-large-v2`** | **Large** | `int8_float16` | Distil-Whisper | 756M | **9.52%** | 0.0% | 0.22s | **50.0x** | `en` (0.984) | 22 |
| 19 | **`distil-large-v2`** | **Large** | `int8` | Distil-Whisper | 756M | **9.52%** | 0.0% | 0.225s | **48.9x** | `en` (0.984) | 22 |
| 20 | **`small`** | **Small** | `float16` | Whisper Standard | 244M | **9.52%** | 0.0% | 0.227s | **48.5x** | `en` (0.963) | 22 |
| 21 | **`small`** | **Small** | `int8_float16` | Whisper Standard | 244M | **9.52%** | 0.0% | 0.23s | **47.9x** | `en` (0.94) | 22 |
| 22 | **`base`** | **Base** | `float16` | Whisper Standard | 74M | **9.52%** | 0.0% | 0.235s | **46.8x** | `en` (0.928) | 22 |
| 23 | **`distil-large-v3`** | **Large** | `int8` | Distil-Whisper | 756M | **9.52%** | 0.0% | 0.235s | **46.7x** | `en` (0.977) | 22 |
| 24 | **`turbo`** | **Large** | `float16` | Whisper Turbo | 809M | **9.52%** | 0.0% | 0.24s | **45.9x** | `en` (0.959) | 22 |
| 25 | **`distil-large-v2`** | **Large** | `float16` | Distil-Whisper | 756M | **9.52%** | 0.0% | 0.242s | **45.5x** | `en` (0.986) | 22 |
| 26 | **`turbo`** | **Large** | `int8_float16` | Whisper Turbo | 809M | **9.52%** | 0.0% | 0.242s | **45.4x** | `en` (0.953) | 22 |
| 27 | **`small`** | **Small** | `int8` | Whisper Standard | 244M | **9.52%** | 0.0% | 0.242s | **45.4x** | `en` (0.94) | 22 |
| 28 | **`turbo`** | **Large** | `int8` | Whisper Turbo | 809M | **9.52%** | 0.0% | 0.244s | **45.1x** | `en` (0.953) | 22 |
| 29 | **`distil-large-v3`** | **Large** | `int8_float16` | Distil-Whisper | 756M | **9.52%** | 0.0% | 0.246s | **44.8x** | `en` (0.977) | 22 |
| 30 | **`distil-large-v3`** | **Large** | `float16` | Distil-Whisper | 756M | **9.52%** | 0.0% | 0.248s | **44.3x** | `en` (0.975) | 22 |
| 31 | **`cahya-whisper-medium-id`** | **Medium** | `float16` | Indonesian Fine-tune | 769M | **9.52%** | 0.0% | 0.278s | **39.6x** | `id` (1) | 22 |
| 32 | **`cahya-whisper-medium-id`** | **Medium** | `int8` | Indonesian Fine-tune | 769M | **9.52%** | 0.0% | 0.279s | **39.4x** | `id` (1) | 22 |
| 33 | **`medium.en`** | **Medium** | `float16` | Whisper English | 769M | **9.52%** | 0.0% | 0.284s | **38.7x** | `en` (1) | 22 |
| 34 | **`medium.en`** | **Medium** | `int8` | Whisper English | 769M | **9.52%** | 0.0% | 0.294s | **37.4x** | `en` (1) | 22 |
| 35 | **`cahya-whisper-medium-id`** | **Medium** | `int8_float16` | Indonesian Fine-tune | 769M | **9.52%** | 0.0% | 0.296s | **37.2x** | `id` (1) | 22 |
| 36 | **`medium`** | **Medium** | `float16` | Whisper Standard | 769M | **9.52%** | 0.0% | 0.296s | **37.1x** | `en` (0.956) | 22 |
| 37 | **`medium.en`** | **Medium** | `int8_float16` | Whisper English | 769M | **9.52%** | 0.0% | 0.306s | **36.0x** | `en` (1) | 22 |
| 38 | **`medium`** | **Medium** | `int8` | Whisper Standard | 769M | **9.52%** | 0.0% | 0.311s | **35.4x** | `en` (0.945) | 22 |
| 39 | **`medium`** | **Medium** | `int8_float16` | Whisper Standard | 769M | **9.52%** | 0.0% | 0.312s | **35.3x** | `en` (0.945) | 22 |
| 40 | **`large-v1`** | **Large** | `int8_float16` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.396s | **27.8x** | `en` (0.964) | 22 |
| 41 | **`large-v2`** | **Large** | `int8_float16` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.397s | **27.7x** | `en` (0.972) | 22 |
| 42 | **`large-v2`** | **Large** | `float16` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.4s | **27.5x** | `en` (0.971) | 22 |
| 43 | **`large-v1`** | **Large** | `float16` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.404s | **27.3x** | `en` (0.962) | 22 |
| 44 | **`large-v3`** | **Large** | `float16` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.412s | **26.7x** | `en` (0.903) | 22 |
| 45 | **`large-v2`** | **Large** | `int8` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.415s | **26.5x** | `en` (0.972) | 22 |
| 46 | **`large-v3`** | **Large** | `int8_float16` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.431s | **25.5x** | `en` (0.887) | 22 |
| 47 | **`large-v3`** | **Large** | `int8` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.442s | **24.9x** | `en` (0.887) | 22 |
| 48 | **`large-v1`** | **Large** | `int8` | Whisper Standard | 1550M | **9.52%** | 0.0% | 0.455s | **24.2x** | `en` (0.964) | 22 |
| 49 | **`distil-small.en`** | **Small** | `int8_float16` | Distil-Whisper | 166M | **28.57%** | 25.3% | 0.149s | **73.9x** | `en` (1) | 15 |
| 50 | **`distil-small.en`** | **Small** | `int8` | Distil-Whisper | 166M | **28.57%** | 25.3% | 0.151s | **72.7x** | `en` (1) | 15 |
| 51 | **`distil-small.en`** | **Small** | `float16` | Distil-Whisper | 166M | **28.57%** | 25.3% | 0.153s | **71.7x** | `en` (1) | 15 |
| 52 | **`cahya-whisper-small-id`** | **Small** | `int8_float16` | Indonesian Fine-tune | 244M | **85.71%** | 90.36% | 0.439s | **25.0x** | `id` (1) | 36 |
| 53 | **`cahya-whisper-small-id`** | **Small** | `int8` | Indonesian Fine-tune | 244M | **85.71%** | 90.36% | 0.443s | **24.8x** | `id` (1) | 36 |
| 54 | **`cahya-whisper-tiny-id`** | **Tiny** | `float16` | Indonesian Fine-tune | 39M | **90.48%** | 74.7% | 0.233s | **47.2x** | `id` (1) | 30 |
| 55 | **`cahya-whisper-tiny-id`** | **Tiny** | `int8_float16` | Indonesian Fine-tune | 39M | **104.76%** | 93.98% | 0.262s | **42.0x** | `id` (1) | 34 |
| 56 | **`cahya-whisper-tiny-id`** | **Tiny** | `int8` | Indonesian Fine-tune | 39M | **104.76%** | 93.98% | 0.275s | **40.0x** | `id` (1) | 34 |
| 57 | **`cahya-whisper-small-id`** | **Small** | `float16` | Indonesian Fine-tune | 244M | **119.05%** | 140.96% | 0.265s | **41.6x** | `id` (1) | 25 |

---

## 2. Performance Breakdown by Size Tier

### Tier: Tiny
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tiny` | `int8_float16` | **9.52%** | **69.2x** | 0.159s | `en` |
| `tiny` | `int8` | **9.52%** | **68.5x** | 0.161s | `en` |
| `tiny` | `float16` | **9.52%** | **68.3x** | 0.161s | `en` |
| `tiny.en` | `int8` | **9.52%** | **63.8x** | 0.172s | `en` |
| `tiny.en` | `int8_float16` | **9.52%** | **61.4x** | 0.179s | `en` |
| `tiny.en` | `float16` | **9.52%** | **56.7x** | 0.194s | `en` |
| `cahya-whisper-tiny-id` | `float16` | **90.48%** | **47.2x** | 0.233s | `id` |
| `cahya-whisper-tiny-id` | `int8_float16` | **104.76%** | **42.0x** | 0.262s | `id` |
| `cahya-whisper-tiny-id` | `int8` | **104.76%** | **40.0x** | 0.275s | `id` |

### Tier: Base
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `base.en` | `float16` | **9.52%** | **64.3x** | 0.171s | `en` |
| `base.en` | `int8_float16` | **9.52%** | **62.4x** | 0.176s | `en` |
| `base` | `int8_float16` | **9.52%** | **58.8x** | 0.187s | `en` |
| `base` | `int8` | **9.52%** | **53.3x** | 0.206s | `en` |
| `base.en` | `int8` | **9.52%** | **52.5x** | 0.21s | `en` |
| `base` | `float16` | **9.52%** | **46.8x** | 0.235s | `en` |

### Tier: Small
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `small.en` | `int8` | **9.52%** | **53.0x** | 0.207s | `en` |
| `small.en` | `float16` | **9.52%** | **51.3x** | 0.214s | `en` |
| `small.en` | `int8_float16` | **9.52%** | **51.0x** | 0.216s | `en` |
| `small` | `float16` | **9.52%** | **48.5x** | 0.227s | `en` |
| `small` | `int8_float16` | **9.52%** | **47.9x** | 0.23s | `en` |
| `small` | `int8` | **9.52%** | **45.4x** | 0.242s | `en` |
| `distil-small.en` | `int8_float16` | **28.57%** | **73.9x** | 0.149s | `en` |
| `distil-small.en` | `int8` | **28.57%** | **72.7x** | 0.151s | `en` |
| `distil-small.en` | `float16` | **28.57%** | **71.7x** | 0.153s | `en` |
| `cahya-whisper-small-id` | `int8_float16` | **85.71%** | **25.0x** | 0.439s | `id` |
| `cahya-whisper-small-id` | `int8` | **85.71%** | **24.8x** | 0.443s | `id` |
| `cahya-whisper-small-id` | `float16` | **119.05%** | **41.6x** | 0.265s | `id` |

### Tier: Medium
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `distil-medium.en` | `int8` | **9.52%** | **65.6x** | 0.168s | `en` |
| `distil-medium.en` | `float16` | **9.52%** | **57.9x** | 0.19s | `en` |
| `distil-medium.en` | `int8_float16` | **9.52%** | **56.8x** | 0.194s | `en` |
| `cahya-whisper-medium-id` | `float16` | **9.52%** | **39.6x** | 0.278s | `id` |
| `cahya-whisper-medium-id` | `int8` | **9.52%** | **39.4x** | 0.279s | `id` |
| `medium.en` | `float16` | **9.52%** | **38.7x** | 0.284s | `en` |
| `medium.en` | `int8` | **9.52%** | **37.4x** | 0.294s | `en` |
| `cahya-whisper-medium-id` | `int8_float16` | **9.52%** | **37.2x** | 0.296s | `id` |
| `medium` | `float16` | **9.52%** | **37.1x** | 0.296s | `en` |
| `medium.en` | `int8_float16` | **9.52%** | **36.0x** | 0.306s | `en` |
| `medium` | `int8` | **9.52%** | **35.4x** | 0.311s | `en` |
| `medium` | `int8_float16` | **9.52%** | **35.3x** | 0.312s | `en` |

### Tier: Large
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `distil-large-v2` | `int8_float16` | **9.52%** | **50.0x** | 0.22s | `en` |
| `distil-large-v2` | `int8` | **9.52%** | **48.9x** | 0.225s | `en` |
| `distil-large-v3` | `int8` | **9.52%** | **46.7x** | 0.235s | `en` |
| `turbo` | `float16` | **9.52%** | **45.9x** | 0.24s | `en` |
| `distil-large-v2` | `float16` | **9.52%** | **45.5x** | 0.242s | `en` |
| `turbo` | `int8_float16` | **9.52%** | **45.4x** | 0.242s | `en` |
| `turbo` | `int8` | **9.52%** | **45.1x** | 0.244s | `en` |
| `distil-large-v3` | `int8_float16` | **9.52%** | **44.8x** | 0.246s | `en` |
| `distil-large-v3` | `float16` | **9.52%** | **44.3x** | 0.248s | `en` |
| `large-v1` | `int8_float16` | **9.52%** | **27.8x** | 0.396s | `en` |
| `large-v2` | `int8_float16` | **9.52%** | **27.7x** | 0.397s | `en` |
| `large-v2` | `float16` | **9.52%** | **27.5x** | 0.4s | `en` |
| `large-v1` | `float16` | **9.52%** | **27.3x** | 0.404s | `en` |
| `large-v3` | `float16` | **9.52%** | **26.7x** | 0.412s | `en` |
| `large-v2` | `int8` | **9.52%** | **26.5x** | 0.415s | `en` |
| `large-v3` | `int8_float16` | **9.52%** | **25.5x** | 0.431s | `en` |
| `large-v3` | `int8` | **9.52%** | **24.9x** | 0.442s | `en` |
| `large-v1` | `int8` | **9.52%** | **24.2x** | 0.455s | `en` |

---

## 3. Full Transcripts Generated per Model Variant

### `tiny` [float16] (Tiny — 39M)
- **Speed**: 68.3x RTF (0.161s) | **WER**: 9.52% | **Language**: `en` (prob: 0.972) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

### `tiny.en` [float16] (Tiny — 39M)
- **Speed**: 56.7x RTF (0.194s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
```

### `cahya-whisper-tiny-id` [float16] (Tiny — 39M)
- **Speed**: 47.2x RTF (0.233s) | **WER**: 90.48% | **Language**: `id` (prob: 1) | **Words**: 30
```text
And, so, my fellow amercks, ASNoth, wajah country, can do for you. AS wajah can do for your country. So, my fellow amercks, ASNoth, wajah country, can do for you.
```

### `base` [float16] (Base — 74M)
- **Speed**: 46.8x RTF (0.235s) | **WER**: 9.52% | **Language**: `en` (prob: 0.928) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `base.en` [float16] (Base — 74M)
- **Speed**: 64.3x RTF (0.171s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `small` [float16] (Small — 244M)
- **Speed**: 48.5x RTF (0.227s) | **WER**: 9.52% | **Language**: `en` (prob: 0.963) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `small.en` [float16] (Small — 244M)
- **Speed**: 51.3x RTF (0.214s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

### `distil-small.en` [float16] (Small — 166M)
- **Speed**: 71.7x RTF (0.153s) | **WER**: 28.57% | **Language**: `en` (prob: 1) | **Words**: 15
```text
And so my fellow Americans, ask not what your country can do for your country.
```

### `cahya-whisper-small-id` [float16] (Small — 244M)
- **Speed**: 41.6x RTF (0.265s) | **WER**: 119.05% | **Language**: `id` (prob: 1) | **Words**: 25
```text
Dan seolahku seorang amerika, meminta bukanlah yang negaramu dapat melakukan untuk kamu, meminta bukanlah yang kamu dapat melakukan untuk kamu. Beritahukan ialah ialah ialah pria.
```

### `medium` [float16] (Medium — 769M)
- **Speed**: 37.1x RTF (0.296s) | **WER**: 9.52% | **Language**: `en` (prob: 0.956) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `medium.en` [float16] (Medium — 769M)
- **Speed**: 38.7x RTF (0.284s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `distil-medium.en` [float16] (Medium — 394M)
- **Speed**: 57.9x RTF (0.19s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

### `cahya-whisper-medium-id` [float16] (Medium — 769M)
- **Speed**: 39.6x RTF (0.278s) | **WER**: 9.52% | **Language**: `id` (prob: 1) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v1` [float16] (Large — 1550M)
- **Speed**: 27.3x RTF (0.404s) | **WER**: 9.52% | **Language**: `en` (prob: 0.962) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v2` [float16] (Large — 1550M)
- **Speed**: 27.5x RTF (0.4s) | **WER**: 9.52% | **Language**: `en` (prob: 0.971) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v3` [float16] (Large — 1550M)
- **Speed**: 26.7x RTF (0.412s) | **WER**: 9.52% | **Language**: `en` (prob: 0.903) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `turbo` [float16] (Large — 809M)
- **Speed**: 45.9x RTF (0.24s) | **WER**: 9.52% | **Language**: `en` (prob: 0.959) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `distil-large-v2` [float16] (Large — 756M)
- **Speed**: 45.5x RTF (0.242s) | **WER**: 9.52% | **Language**: `en` (prob: 0.986) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

### `distil-large-v3` [float16] (Large — 756M)
- **Speed**: 44.3x RTF (0.248s) | **WER**: 9.52% | **Language**: `en` (prob: 0.975) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

### `tiny` [int8_float16] (Tiny — 39M)
- **Speed**: 69.2x RTF (0.159s) | **WER**: 9.52% | **Language**: `en` (prob: 0.973) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

### `tiny.en` [int8_float16] (Tiny — 39M)
- **Speed**: 61.4x RTF (0.179s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
```

### `cahya-whisper-tiny-id` [int8_float16] (Tiny — 39M)
- **Speed**: 42.0x RTF (0.262s) | **WER**: 104.76% | **Language**: `id` (prob: 1) | **Words**: 34
```text
And, so, my fellow amercks, ASNoth, wajah country, can do for you. AS wajah can do for your country. So, my fellow amercks, ASNoth, wajah country, can do for your country. AS wajah country.
```

### `base` [int8_float16] (Base — 74M)
- **Speed**: 58.8x RTF (0.187s) | **WER**: 9.52% | **Language**: `en` (prob: 0.928) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `base.en` [int8_float16] (Base — 74M)
- **Speed**: 62.4x RTF (0.176s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `small` [int8_float16] (Small — 244M)
- **Speed**: 47.9x RTF (0.23s) | **WER**: 9.52% | **Language**: `en` (prob: 0.94) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `small.en` [int8_float16] (Small — 244M)
- **Speed**: 51.0x RTF (0.216s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `distil-small.en` [int8_float16] (Small — 166M)
- **Speed**: 73.9x RTF (0.149s) | **WER**: 28.57% | **Language**: `en` (prob: 1) | **Words**: 15
```text
And so my fellow Americans, ask not what your country can do for your country.
```

### `cahya-whisper-small-id` [int8_float16] (Small — 244M)
- **Speed**: 25.0x RTF (0.439s) | **WER**: 85.71% | **Language**: `id` (prob: 1) | **Words**: 36
```text
Dan so, my fellow American, ask not what your country can do for you, ask what you can do for your country! Bagaimana kalau kita pergi ke kolam renang. Bagaimana kalau kita pergi ke kolam renang.
```

### `medium` [int8_float16] (Medium — 769M)
- **Speed**: 35.3x RTF (0.312s) | **WER**: 9.52% | **Language**: `en` (prob: 0.945) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `medium.en` [int8_float16] (Medium — 769M)
- **Speed**: 36.0x RTF (0.306s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `distil-medium.en` [int8_float16] (Medium — 394M)
- **Speed**: 56.8x RTF (0.194s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

### `cahya-whisper-medium-id` [int8_float16] (Medium — 769M)
- **Speed**: 37.2x RTF (0.296s) | **WER**: 9.52% | **Language**: `id` (prob: 1) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v1` [int8_float16] (Large — 1550M)
- **Speed**: 27.8x RTF (0.396s) | **WER**: 9.52% | **Language**: `en` (prob: 0.964) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v2` [int8_float16] (Large — 1550M)
- **Speed**: 27.7x RTF (0.397s) | **WER**: 9.52% | **Language**: `en` (prob: 0.972) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v3` [int8_float16] (Large — 1550M)
- **Speed**: 25.5x RTF (0.431s) | **WER**: 9.52% | **Language**: `en` (prob: 0.887) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `turbo` [int8_float16] (Large — 809M)
- **Speed**: 45.4x RTF (0.242s) | **WER**: 9.52% | **Language**: `en` (prob: 0.953) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `distil-large-v2` [int8_float16] (Large — 756M)
- **Speed**: 50.0x RTF (0.22s) | **WER**: 9.52% | **Language**: `en` (prob: 0.984) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

### `distil-large-v3` [int8_float16] (Large — 756M)
- **Speed**: 44.8x RTF (0.246s) | **WER**: 9.52% | **Language**: `en` (prob: 0.977) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

### `tiny` [int8] (Tiny — 39M)
- **Speed**: 68.5x RTF (0.161s) | **WER**: 9.52% | **Language**: `en` (prob: 0.973) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

### `tiny.en` [int8] (Tiny — 39M)
- **Speed**: 63.8x RTF (0.172s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
```

### `cahya-whisper-tiny-id` [int8] (Tiny — 39M)
- **Speed**: 40.0x RTF (0.275s) | **WER**: 104.76% | **Language**: `id` (prob: 1) | **Words**: 34
```text
And, so, my fellow amercks, ASNoth, wajah country, can do for you. AS wajah can do for your country. So, my fellow amercks, ASNoth, wajah country, can do for your country. AS wajah country.
```

### `base` [int8] (Base — 74M)
- **Speed**: 53.3x RTF (0.206s) | **WER**: 9.52% | **Language**: `en` (prob: 0.928) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `base.en` [int8] (Base — 74M)
- **Speed**: 52.5x RTF (0.21s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `small` [int8] (Small — 244M)
- **Speed**: 45.4x RTF (0.242s) | **WER**: 9.52% | **Language**: `en` (prob: 0.94) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `small.en` [int8] (Small — 244M)
- **Speed**: 53.0x RTF (0.207s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `distil-small.en` [int8] (Small — 166M)
- **Speed**: 72.7x RTF (0.151s) | **WER**: 28.57% | **Language**: `en` (prob: 1) | **Words**: 15
```text
And so my fellow Americans, ask not what your country can do for your country.
```

### `cahya-whisper-small-id` [int8] (Small — 244M)
- **Speed**: 24.8x RTF (0.443s) | **WER**: 85.71% | **Language**: `id` (prob: 1) | **Words**: 36
```text
Dan so, my fellow American, ask not what your country can do for you, ask what you can do for your country! Bagaimana kalau kita pergi ke kolam renang. Bagaimana kalau kita pergi ke kolam renang.
```

### `medium` [int8] (Medium — 769M)
- **Speed**: 35.4x RTF (0.311s) | **WER**: 9.52% | **Language**: `en` (prob: 0.945) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `medium.en` [int8] (Medium — 769M)
- **Speed**: 37.4x RTF (0.294s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `distil-medium.en` [int8] (Medium — 394M)
- **Speed**: 65.6x RTF (0.168s) | **WER**: 9.52% | **Language**: `en` (prob: 1) | **Words**: 22
```text
And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

### `cahya-whisper-medium-id` [int8] (Medium — 769M)
- **Speed**: 39.4x RTF (0.279s) | **WER**: 9.52% | **Language**: `id` (prob: 1) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v1` [int8] (Large — 1550M)
- **Speed**: 24.2x RTF (0.455s) | **WER**: 9.52% | **Language**: `en` (prob: 0.964) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v2` [int8] (Large — 1550M)
- **Speed**: 26.5x RTF (0.415s) | **WER**: 9.52% | **Language**: `en` (prob: 0.972) | **Words**: 22
```text
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `large-v3` [int8] (Large — 1550M)
- **Speed**: 24.9x RTF (0.442s) | **WER**: 9.52% | **Language**: `en` (prob: 0.887) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `turbo` [int8] (Large — 809M)
- **Speed**: 45.1x RTF (0.244s) | **WER**: 9.52% | **Language**: `en` (prob: 0.953) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

### `distil-large-v2` [int8] (Large — 756M)
- **Speed**: 48.9x RTF (0.225s) | **WER**: 9.52% | **Language**: `en` (prob: 0.984) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

### `distil-large-v3` [int8] (Large — 756M)
- **Speed**: 46.7x RTF (0.235s) | **WER**: 9.52% | **Language**: `en` (prob: 0.977) | **Words**: 22
```text
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```
