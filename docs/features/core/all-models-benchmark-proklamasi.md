# Comprehensive Benchmark Report: `proklamasi.wav`

> **Hardware**: NVIDIA L40  
> **Audio Sample**: `proklamasi.wav` (Duration: 48.52s)  
> **Date**: 2026-08-27 20:46:19  
> **Engine**: Faster-Whisper (CTranslate2) across 3 Quantization Modes

> **Ground Truth Reference**: "Kami bangsa Indonesia dengan ini menjatakan kemerdekaan Indonesia. Hal-hal jang mengenai pemindahan kekoeasaan d.l.l., diselenggarakan dengan tjara saksama dan dalam tempo jang sesingkat-singkatnja. Djakarta, hari 17 boelan 8 tahoen 05. Atas nama bangsa Indonesia, Soekarno Hatta."

---

## 1. Summary Comparison Matrix

| Rank | Model | Size Tier | Quantization | Family | Params | WER (%) | CER (%) | Time (s) | Speed (x RTF) | Lang (Prob) | Words |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **`turbo`** | **Large** | `int8` | Whisper Turbo | 809M | **41.67%** | 17.17% | 0.49s | **99.1x** | `id` (0.995) | 34 |
| 2 | **`turbo`** | **Large** | `int8_float16` | Whisper Turbo | 809M | **41.67%** | 17.17% | 0.502s | **96.6x** | `id` (0.995) | 34 |
| 3 | **`cahya-whisper-small-id`** | **Small** | `float16` | Indonesian Fine-tune | 244M | **44.44%** | 18.03% | 0.497s | **97.6x** | `id` (1) | 34 |
| 4 | **`cahya-whisper-small-id`** | **Small** | `int8` | Indonesian Fine-tune | 244M | **44.44%** | 18.03% | 0.504s | **96.3x** | `id` (1) | 34 |
| 5 | **`cahya-whisper-small-id`** | **Small** | `int8_float16` | Indonesian Fine-tune | 244M | **44.44%** | 18.03% | 0.511s | **95.0x** | `id` (1) | 34 |
| 6 | **`turbo`** | **Large** | `float16` | Whisper Turbo | 809M | **47.22%** | 17.17% | 0.505s | **96.2x** | `id` (0.996) | 33 |
| 7 | **`medium`** | **Medium** | `float16` | Whisper Standard | 769M | **47.22%** | 18.88% | 0.718s | **67.6x** | `id` (0.838) | 34 |
| 8 | **`medium`** | **Medium** | `int8` | Whisper Standard | 769M | **47.22%** | 18.88% | 0.753s | **64.4x** | `id` (0.809) | 34 |
| 9 | **`medium`** | **Medium** | `int8_float16` | Whisper Standard | 769M | **47.22%** | 18.88% | 0.785s | **61.8x** | `id` (0.809) | 34 |
| 10 | **`large-v2`** | **Large** | `float16` | Whisper Standard | 1550M | **47.22%** | 18.03% | 0.973s | **49.9x** | `id` (0.755) | 33 |
| 11 | **`large-v3`** | **Large** | `int8` | Whisper Standard | 1550M | **47.22%** | 18.03% | 0.973s | **49.9x** | `id` (0.743) | 33 |
| 12 | **`large-v2`** | **Large** | `int8_float16` | Whisper Standard | 1550M | **47.22%** | 17.17% | 0.974s | **49.8x** | `id` (0.77) | 33 |
| 13 | **`large-v2`** | **Large** | `int8` | Whisper Standard | 1550M | **47.22%** | 17.17% | 0.984s | **49.3x** | `id` (0.77) | 33 |
| 14 | **`large-v3`** | **Large** | `int8_float16` | Whisper Standard | 1550M | **47.22%** | 18.03% | 0.994s | **48.8x** | `id` (0.743) | 33 |
| 15 | **`large-v3`** | **Large** | `float16` | Whisper Standard | 1550M | **47.22%** | 17.17% | 1.007s | **48.2x** | `id` (0.776) | 33 |
| 16 | **`small`** | **Small** | `int8_float16` | Whisper Standard | 244M | **50.0%** | 18.88% | 0.48s | **101.2x** | `id` (0.627) | 34 |
| 17 | **`small`** | **Small** | `float16` | Whisper Standard | 244M | **50.0%** | 18.88% | 0.496s | **97.9x** | `id` (0.691) | 34 |
| 18 | **`small`** | **Small** | `int8` | Whisper Standard | 244M | **50.0%** | 18.88% | 0.495s | **97.9x** | `id` (0.627) | 34 |
| 19 | **`large-v1`** | **Large** | `float16` | Whisper Standard | 1550M | **50.0%** | 18.88% | 1.018s | **47.7x** | `id` (0.782) | 34 |
| 20 | **`large-v1`** | **Large** | `int8` | Whisper Standard | 1550M | **50.0%** | 18.88% | 1.029s | **47.2x** | `id` (0.828) | 34 |
| 21 | **`large-v1`** | **Large** | `int8_float16` | Whisper Standard | 1550M | **50.0%** | 18.88% | 1.032s | **47.0x** | `id` (0.828) | 34 |
| 22 | **`cahya-whisper-medium-id`** | **Medium** | `float16` | Indonesian Fine-tune | 769M | **55.56%** | 33.05% | 0.621s | **78.2x** | `id` (1) | 29 |
| 23 | **`cahya-whisper-medium-id`** | **Medium** | `int8_float16` | Indonesian Fine-tune | 769M | **55.56%** | 33.05% | 0.642s | **75.6x** | `id` (1) | 29 |
| 24 | **`cahya-whisper-medium-id`** | **Medium** | `int8` | Indonesian Fine-tune | 769M | **55.56%** | 33.05% | 0.674s | **72.0x** | `id` (1) | 29 |
| 25 | **`base`** | **Base** | `int8` | Whisper Standard | 74M | **72.22%** | 24.89% | 0.404s | **120.1x** | `id` (0.965) | 36 |
| 26 | **`base`** | **Base** | `float16` | Whisper Standard | 74M | **72.22%** | 26.61% | 0.412s | **117.8x** | `id` (0.957) | 37 |
| 27 | **`base`** | **Base** | `int8_float16` | Whisper Standard | 74M | **72.22%** | 24.89% | 0.426s | **113.9x** | `id` (0.965) | 36 |
| 28 | **`cahya-whisper-tiny-id`** | **Tiny** | `float16` | Indonesian Fine-tune | 39M | **83.33%** | 36.05% | 0.358s | **135.5x** | `id` (1) | 32 |
| 29 | **`cahya-whisper-tiny-id`** | **Tiny** | `int8_float16` | Indonesian Fine-tune | 39M | **83.33%** | 36.05% | 0.555s | **87.5x** | `id` (1) | 32 |
| 30 | **`tiny`** | **Tiny** | `float16` | Whisper Standard | 39M | **83.33%** | 33.48% | 0.711s | **68.2x** | `id` (0.492) | 41 |
| 31 | **`distil-large-v3`** | **Large** | `int8` | Distil-Whisper | 756M | **86.11%** | 59.66% | 0.739s | **65.7x** | `en` (0.619) | 36 |
| 32 | **`distil-large-v3`** | **Large** | `int8_float16` | Distil-Whisper | 756M | **86.11%** | 59.66% | 0.748s | **64.8x** | `en` (0.619) | 36 |
| 33 | **`tiny`** | **Tiny** | `int8` | Whisper Standard | 39M | **88.89%** | 33.05% | 0.426s | **113.8x** | `id` (0.49) | 42 |
| 34 | **`tiny`** | **Tiny** | `int8_float16` | Whisper Standard | 39M | **88.89%** | 33.05% | 0.451s | **107.6x** | `id` (0.49) | 42 |
| 35 | **`distil-large-v2`** | **Large** | `int8_float16` | Distil-Whisper | 756M | **94.44%** | 73.82% | 0.552s | **88.0x** | `en` (0.509) | 22 |
| 36 | **`distil-large-v2`** | **Large** | `int8` | Distil-Whisper | 756M | **94.44%** | 73.82% | 0.558s | **87.0x** | `en` (0.509) | 22 |
| 37 | **`distil-small.en`** | **Small** | `int8` | Distil-Whisper | 166M | **94.44%** | 75.54% | 0.617s | **78.7x** | `en` (1) | 11 |
| 38 | **`distil-small.en`** | **Small** | `float16` | Distil-Whisper | 166M | **94.44%** | 61.37% | 0.815s | **59.5x** | `en` (1) | 14 |
| 39 | **`distil-large-v3`** | **Large** | `float16` | Distil-Whisper | 756M | **94.44%** | 56.22% | 0.815s | **59.5x** | `en` (0.605) | 40 |
| 40 | **`distil-small.en`** | **Small** | `int8_float16` | Distil-Whisper | 166M | **94.44%** | 106.87% | 2.048s | **23.7x** | `en` (1) | 16 |
| 41 | **`distil-medium.en`** | **Medium** | `int8` | Distil-Whisper | 394M | **97.22%** | 56.65% | 0.836s | **58.0x** | `en` (1) | 25 |
| 42 | **`distil-medium.en`** | **Medium** | `int8_float16` | Distil-Whisper | 394M | **97.22%** | 56.65% | 0.838s | **57.9x** | `en` (1) | 25 |
| 43 | **`distil-medium.en`** | **Medium** | `float16` | Distil-Whisper | 394M | **97.22%** | 53.65% | 0.892s | **54.4x** | `en` (1) | 28 |
| 44 | **`small.en`** | **Small** | `int8` | Whisper English | 244M | **100.0%** | 43.35% | 0.633s | **76.7x** | `en` (1) | 43 |
| 45 | **`small.en`** | **Small** | `float16` | Whisper English | 244M | **100.0%** | 43.35% | 0.634s | **76.5x** | `en` (1) | 44 |
| 46 | **`small.en`** | **Small** | `int8_float16` | Whisper English | 244M | **100.0%** | 43.35% | 0.658s | **73.8x** | `en` (1) | 43 |
| 47 | **`tiny.en`** | **Tiny** | `int8_float16` | Whisper English | 39M | **100.0%** | 100.0% | 0.824s | **58.8x** | `en` (1) | 0 |
| 48 | **`tiny.en`** | **Tiny** | `float16` | Whisper English | 39M | **100.0%** | 100.0% | 1.147s | **42.3x** | `en` (1) | 0 |
| 49 | **`tiny.en`** | **Tiny** | `int8` | Whisper English | 39M | **100.0%** | 100.0% | 1.446s | **33.5x** | `en` (1) | 0 |
| 50 | **`medium.en`** | **Medium** | `float16` | Whisper English | 769M | **100.0%** | 39.48% | 1.76s | **27.6x** | `en` (1) | 42 |
| 51 | **`base.en`** | **Base** | `float16` | Whisper English | 74M | **102.78%** | 49.36% | 0.478s | **101.5x** | `en` (1) | 42 |
| 52 | **`distil-large-v2`** | **Large** | `float16` | Distil-Whisper | 756M | **105.56%** | 70.39% | 0.609s | **79.6x** | `en` (0.512) | 40 |
| 53 | **`medium.en`** | **Medium** | `int8` | Whisper English | 769M | **105.56%** | 40.34% | 0.994s | **48.8x** | `en` (1) | 44 |
| 54 | **`medium.en`** | **Medium** | `int8_float16` | Whisper English | 769M | **105.56%** | 40.34% | 0.999s | **48.6x** | `en` (1) | 44 |
| 55 | **`base.en`** | **Base** | `int8` | Whisper English | 74M | **119.44%** | 49.36% | 0.47s | **103.2x** | `en` (1) | 48 |
| 56 | **`base.en`** | **Base** | `int8_float16` | Whisper English | 74M | **119.44%** | 49.36% | 0.475s | **102.3x** | `en` (1) | 48 |
| 57 | **`cahya-whisper-tiny-id`** | **Tiny** | `int8` | Indonesian Fine-tune | 39M | **130.56%** | 75.97% | 4.171s | **11.6x** | `id` (1) | 54 |

---

## 2. Performance Breakdown by Size Tier

### Tier: Tiny
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cahya-whisper-tiny-id` | `float16` | **83.33%** | **135.5x** | 0.358s | `id` |
| `cahya-whisper-tiny-id` | `int8_float16` | **83.33%** | **87.5x** | 0.555s | `id` |
| `tiny` | `float16` | **83.33%** | **68.2x** | 0.711s | `id` |
| `tiny` | `int8` | **88.89%** | **113.8x** | 0.426s | `id` |
| `tiny` | `int8_float16` | **88.89%** | **107.6x** | 0.451s | `id` |
| `tiny.en` | `int8_float16` | **100.0%** | **58.8x** | 0.824s | `en` |
| `tiny.en` | `float16` | **100.0%** | **42.3x** | 1.147s | `en` |
| `tiny.en` | `int8` | **100.0%** | **33.5x** | 1.446s | `en` |
| `cahya-whisper-tiny-id` | `int8` | **130.56%** | **11.6x** | 4.171s | `id` |

### Tier: Base
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `base` | `int8` | **72.22%** | **120.1x** | 0.404s | `id` |
| `base` | `float16` | **72.22%** | **117.8x** | 0.412s | `id` |
| `base` | `int8_float16` | **72.22%** | **113.9x** | 0.426s | `id` |
| `base.en` | `float16` | **102.78%** | **101.5x** | 0.478s | `en` |
| `base.en` | `int8` | **119.44%** | **103.2x** | 0.47s | `en` |
| `base.en` | `int8_float16` | **119.44%** | **102.3x** | 0.475s | `en` |

### Tier: Small
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cahya-whisper-small-id` | `float16` | **44.44%** | **97.6x** | 0.497s | `id` |
| `cahya-whisper-small-id` | `int8` | **44.44%** | **96.3x** | 0.504s | `id` |
| `cahya-whisper-small-id` | `int8_float16` | **44.44%** | **95.0x** | 0.511s | `id` |
| `small` | `int8_float16` | **50.0%** | **101.2x** | 0.48s | `id` |
| `small` | `float16` | **50.0%** | **97.9x** | 0.496s | `id` |
| `small` | `int8` | **50.0%** | **97.9x** | 0.495s | `id` |
| `distil-small.en` | `int8` | **94.44%** | **78.7x** | 0.617s | `en` |
| `distil-small.en` | `float16` | **94.44%** | **59.5x** | 0.815s | `en` |
| `distil-small.en` | `int8_float16` | **94.44%** | **23.7x** | 2.048s | `en` |
| `small.en` | `int8` | **100.0%** | **76.7x** | 0.633s | `en` |
| `small.en` | `float16` | **100.0%** | **76.5x** | 0.634s | `en` |
| `small.en` | `int8_float16` | **100.0%** | **73.8x** | 0.658s | `en` |

### Tier: Medium
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `medium` | `float16` | **47.22%** | **67.6x** | 0.718s | `id` |
| `medium` | `int8` | **47.22%** | **64.4x** | 0.753s | `id` |
| `medium` | `int8_float16` | **47.22%** | **61.8x** | 0.785s | `id` |
| `cahya-whisper-medium-id` | `float16` | **55.56%** | **78.2x** | 0.621s | `id` |
| `cahya-whisper-medium-id` | `int8_float16` | **55.56%** | **75.6x** | 0.642s | `id` |
| `cahya-whisper-medium-id` | `int8` | **55.56%** | **72.0x** | 0.674s | `id` |
| `distil-medium.en` | `int8` | **97.22%** | **58.0x** | 0.836s | `en` |
| `distil-medium.en` | `int8_float16` | **97.22%** | **57.9x** | 0.838s | `en` |
| `distil-medium.en` | `float16` | **97.22%** | **54.4x** | 0.892s | `en` |
| `medium.en` | `float16` | **100.0%** | **27.6x** | 1.76s | `en` |
| `medium.en` | `int8` | **105.56%** | **48.8x** | 0.994s | `en` |
| `medium.en` | `int8_float16` | **105.56%** | **48.6x** | 0.999s | `en` |

### Tier: Large
| Model | Quant | WER (%) | Speed (x RTF) | Time (s) | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `turbo` | `int8` | **41.67%** | **99.1x** | 0.49s | `id` |
| `turbo` | `int8_float16` | **41.67%** | **96.6x** | 0.502s | `id` |
| `turbo` | `float16` | **47.22%** | **96.2x** | 0.505s | `id` |
| `large-v2` | `float16` | **47.22%** | **49.9x** | 0.973s | `id` |
| `large-v3` | `int8` | **47.22%** | **49.9x** | 0.973s | `id` |
| `large-v2` | `int8_float16` | **47.22%** | **49.8x** | 0.974s | `id` |
| `large-v2` | `int8` | **47.22%** | **49.3x** | 0.984s | `id` |
| `large-v3` | `int8_float16` | **47.22%** | **48.8x** | 0.994s | `id` |
| `large-v3` | `float16` | **47.22%** | **48.2x** | 1.007s | `id` |
| `large-v1` | `float16` | **50.0%** | **47.7x** | 1.018s | `id` |
| `large-v1` | `int8` | **50.0%** | **47.2x** | 1.029s | `id` |
| `large-v1` | `int8_float16` | **50.0%** | **47.0x** | 1.032s | `id` |
| `distil-large-v3` | `int8` | **86.11%** | **65.7x** | 0.739s | `en` |
| `distil-large-v3` | `int8_float16` | **86.11%** | **64.8x** | 0.748s | `en` |
| `distil-large-v2` | `int8_float16` | **94.44%** | **88.0x** | 0.552s | `en` |
| `distil-large-v2` | `int8` | **94.44%** | **87.0x** | 0.558s | `en` |
| `distil-large-v3` | `float16` | **94.44%** | **59.5x** | 0.815s | `en` |
| `distil-large-v2` | `float16` | **105.56%** | **79.6x** | 0.609s | `en` |

---

## 3. Full Transcripts Generated per Model Variant

### `tiny` [float16] (Tiny — 39M)
- **Speed**: 68.2x RTF (0.711s) | **WER**: 83.33% | **Language**: `id` (prob: 0.492) | **Words**: 41
```text
Kau kelamati pangsa Indonesia dengan ini menyatakan kemerdilkaan Indonesia, hal hal yang mengenai pemindahan kekuatan, canlai lain, kisilan garaakan dengan cara saksama, canjalam tempu, yang sesingkat kingkatnya, Jakarta, tujubalat Agustus, teribu 70 ratus ampat bululima, atas nama pangsa Indonesia, cukarno, hata.
```

### `tiny.en` [float16] (Tiny — 39M)
- **Speed**: 42.3x RTF (1.147s) | **WER**: 100.0% | **Language**: `en` (prob: 1) | **Words**: 0
```text

```

### `cahya-whisper-tiny-id` [float16] (Tiny — 39M)
- **Speed**: 135.5x RTF (0.358s) | **WER**: 83.33% | **Language**: `id` (prob: 1) | **Words**: 32
```text
kelamati pangka indonesia dengan ini menyatakan kemerdilkaan indonesia al hal yang mengenai permindahan kekuatan jalan lain-lain di selenggarakan dengan cara saksama jalan dalam tempo yang setingkap tingkatnya jatarkan tujuh balat Agustus, teribu.
```

### `base` [float16] (Base — 74M)
- **Speed**: 117.8x RTF (0.412s) | **WER**: 72.22% | **Language**: `id` (prob: 0.957) | **Words**: 37
```text
Jauh kelama di tangsa Indonesia, dengan ini menyatakan kemerdekaan Indonesia. Al-hal yang mengenai pemindahan kekuatan dan lain-lain di selenggarakan dengan cara saktama, dan dalam tempu yang sesinkap-sinkapnya, Jakarta 17 Agustus, 1945 atas 6 bangsa Indonesia, Tukarno, Atat.
```

### `base.en` [float16] (Base — 74M)
- **Speed**: 101.5x RTF (0.478s) | **WER**: 102.78% | **Language**: `en` (prob: 1) | **Words**: 42
```text
No calamity, pamsa indonesia, tenan yin, manyatakan, kurmertika, and indonesia. Al har yang mananai permin dahankakwataan, chanlai lai, isalangarakan, tenan chara saktama, chanjalam temple, yang sashin kapshin kapnya, yakarta, vujubalat agusto, saribu, sambiram kantra to sampakululima, atas nama pamsa indonesia, tukarno, atas.
```

### `small` [float16] (Small — 244M)
- **Speed**: 97.9x RTF (0.496s) | **WER**: 50.0% | **Language**: `id` (prob: 0.691) | **Words**: 34
```text
proklamasi bangsa Indonesia, dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempoh yang sesingkat-singkatnya, Jakarta 17 Agustus 1945 atas nama bangsa Indonesia, Sukarno Atta.
```

### `small.en` [float16] (Small — 244M)
- **Speed**: 76.5x RTF (0.634s) | **WER**: 100.0% | **Language**: `en` (prob: 1) | **Words**: 44
```text
Rokkala Mahasi, Pangsa, Indonesia, Tengnan Inu, Manyatakan, Karmar Deka, and Indonesia. Hal Hal Yang Manganai, Praminda Han Kukwataan, Tan Lai Laien, Yisalanga Arakan, Tengnan Chara Saksama, Tan Dalam Tempo, Yangtasin Kapsin Kapnya, Yakarta, Hujubalas Agustus, Tareebu Similanratos Ampatululima, Atas Nama Pangsa, Indonesia, Tukarno, Ata.
```

### `distil-small.en` [float16] (Small — 166M)
- **Speed**: 59.5x RTF (0.815s) | **WER**: 94.44% | **Language**: `en` (prob: 1) | **Words**: 14
```text
pro-Kla-ma-shi, Pangsa, Indonesia. Tungan-inu, Maniyatakan, Krumar-deakah, and Indonesia. Al-Hal-Yamangmanai per min-dahan-kakwatha-an, Tan-line-line, E-Sallangarakan, -chara-sakthamah-tham,
```

### `cahya-whisper-small-id` [float16] (Small — 244M)
- **Speed**: 97.6x RTF (0.497s) | **WER**: 44.44% | **Language**: `id` (prob: 1) | **Words**: 34
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia, hal-hal yang mengenai pemindahan kekuasaan, dan lain-lain diselenggarakan dengan cara saksama, dan dalam tempo yang sesingkat-singkatnya. Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno Ata.
```

### `medium` [float16] (Medium — 769M)
- **Speed**: 67.6x RTF (0.718s) | **WER**: 47.22% | **Language**: `id` (prob: 0.838) | **Words**: 34
```text
proklamasi bangsa Indonesia, dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama. Dan dalam tempuh yang sesingkat-singkatnya, Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Sukarno Hatta.
```

### `medium.en` [float16] (Medium — 769M)
- **Speed**: 27.6x RTF (1.76s) | **WER**: 100.0% | **Language**: `en` (prob: 1) | **Words**: 42
```text
proclamati, Bongsa, Indonesia, dungan ini, manyatakan, kumar dekhaan, Indonesia, al hal yang manganai, paminda han kakua saan, tan lain lain, gisalangarakan, dungan chara saktama, tan dalam tempu, yang sisinkat sinkatna, jakarta, sujubalas agustus, saribu similanratos ampatululima, atak nama Bongsa Indonesia, sukarno atak.
```

### `distil-medium.en` [float16] (Medium — 394M)
- **Speed**: 54.4x RTF (0.892s) | **WER**: 97.22% | **Language**: `en` (prob: 1) | **Words**: 28
```text
Proclamati, Pangsa, Indonesia. Tengan I mean Meyatakan, Kermardika and Indonesia. Al-Haulayangman Nain Permina Hankakwaraan, John Lain, Jisalangarakan, Tengan Chana Janaan Sharra Sakamah, Jandalam Tham Thamalam Thamalamalamalamalamu Kymah, ,
```

### `cahya-whisper-medium-id` [float16] (Medium — 769M)
- **Speed**: 78.2x RTF (0.621s) | **WER**: 55.56% | **Language**: `id` (prob: 1) | **Words**: 29
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia, hal-hal yang mengenai pemindahan kekuasaan, dan lain-lain diselenggarakan dengan cara saksama. Beratus empat puluh lima atas nama bangsa Indonesia Soekarno Atta.
```

### `large-v1` [float16] (Large — 1550M)
- **Speed**: 47.7x RTF (1.018s) | **WER**: 50.0% | **Language**: `id` (prob: 0.782) | **Words**: 34
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara seksama dan dalam tempoh yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Sukarno Hatta.
```

### `large-v2` [float16] (Large — 1550M)
- **Speed**: 49.9x RTF (0.973s) | **WER**: 47.22% | **Language**: `id` (prob: 0.755) | **Words**: 33
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Sukarno-Hatta
```

### `large-v3` [float16] (Large — 1550M)
- **Speed**: 48.2x RTF (1.007s) | **WER**: 47.22% | **Language**: `id` (prob: 0.776) | **Words**: 33
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang sesingkat-singkatnya. Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno-Hatta.
```

### `turbo` [float16] (Large — 809M)
- **Speed**: 96.2x RTF (0.505s) | **WER**: 47.22% | **Language**: `id` (prob: 0.996) | **Words**: 33
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno-Hatta.
```

### `distil-large-v2` [float16] (Large — 756M)
- **Speed**: 79.6x RTF (0.609s) | **WER**: 105.56% | **Language**: `en` (prob: 0.512) | **Words**: 40
```text
Proclamation, thanks, Indonesia. With this, it menataken Indonesia. All the things that the mingean, the theareran the theen thear thear the thear, and in thelengar the to the same and in then. And in the temp that I have today.
```

### `distil-large-v3` [float16] (Large — 756M)
- **Speed**: 59.5x RTF (0.815s) | **WER**: 94.44% | **Language**: `en` (prob: 0.605) | **Words**: 40
```text
Proclamation of the people Indonesia With this It's made make Themerdecaan Indonesia All-hall that about Demandahe and other Lain Langeara with Carta and in Tempo which Sinkat Singat Jakarta 17 August 1,09 245 atas Nama name of Indonesia Sukarno Hata
```

### `tiny` [int8_float16] (Tiny — 39M)
- **Speed**: 107.6x RTF (0.451s) | **WER**: 88.89% | **Language**: `id` (prob: 0.49) | **Words**: 42
```text
Kau kelamati pangsa Indonesia dengan ini menyatakan kemerdilkaan Indonesia, hal hal yang mengenai pemindahan kekuatan, canlai lain, di selengah rakan, telgan cara saksama, canjalam tempu, yang sesingkat kingkatnya. Jakarta, tujubalat Agustus, teribu 70 ratus ampat bululima, atas nama pangsa Indonesia, cukarno hata.
```

### `tiny.en` [int8_float16] (Tiny — 39M)
- **Speed**: 58.8x RTF (0.824s) | **WER**: 100.0% | **Language**: `en` (prob: 1) | **Words**: 0
```text

```

### `cahya-whisper-tiny-id` [int8_float16] (Tiny — 39M)
- **Speed**: 87.5x RTF (0.555s) | **WER**: 83.33% | **Language**: `id` (prob: 1) | **Words**: 32
```text
kelamati pangka indonesia dengan ini menyatakan kemerdilkaan indonesia al hal yang mengenai pembindahan kekuatan jalan lain-lain di Selenggarakan dengan cara saksama jalan dalam tempo yang setingkap tingkatnya jatarkan tujuh balat Agustus, teribu.
```

### `base` [int8_float16] (Base — 74M)
- **Speed**: 113.9x RTF (0.426s) | **WER**: 72.22% | **Language**: `id` (prob: 0.965) | **Words**: 36
```text
Jauh kelama di tangsa Indonesia, dengan ini menyatakan kemerdekaan Indonesia. Al-hal yang mengenai pemindahan kekuatan dan lain-lain di selenggarakan dengan cara saksama, dan dalam tempu yang seringkap-tingkapnya, Jakarta, 17 agustus, 1945, atas namapangsa Indonesia, Tukarno, atas.
```

### `base.en` [int8_float16] (Base — 74M)
- **Speed**: 102.3x RTF (0.475s) | **WER**: 119.44% | **Language**: `en` (prob: 1) | **Words**: 48
```text
No calamity, pamsa indonesia, tenan yi, manya takan, kurmertika, and indonesia. Al har yang manna nai permin dahanka kwata an, chan lai nai, isalangara kan, tenan chara saktama, chan dalam tempu, yang satin kapshin kapnya, yakarta, ujubalat argusto, saribu sambirantra to sampakcululima, taktas nama pamsa indonesia, ukarno, taktas.
```

### `small` [int8_float16] (Small — 244M)
- **Speed**: 101.2x RTF (0.48s) | **WER**: 50.0% | **Language**: `id` (prob: 0.627) | **Words**: 34
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempu yang sesingkat-singkatnya. Jakarta 17 Agustus 1945 atas nama bangsa Indonesia, Sukarno Atta.
```

### `small.en` [int8_float16] (Small — 244M)
- **Speed**: 73.8x RTF (0.658s) | **WER**: 100.0% | **Language**: `en` (prob: 1) | **Words**: 43
```text
Rokkala Mahasi, Pangsa, Indonesia, Tengnan Inu, Manyatakan, Karmar Deka, and Indonesia. Hal Hal, Yang Manganai, Praminda Han, Kukwataan, Tan Lai Lai, Yisalanga Arakan, Tengnan Chara Saksama, Tan Dalam Tempo, Yangshasinkap, Tinkapnya, Yakarta, Hujubalas Agustus, Tareebu Similandratos Ampatululima, Atas Nama Pangsa, Indonesia, Tukarno, Ata.
```

### `distil-small.en` [int8_float16] (Small — 166M)
- **Speed**: 23.7x RTF (2.048s) | **WER**: 94.44% | **Language**: `en` (prob: 1) | **Words**: 16
```text
pro-Kla-macy, Pangsa, Indonesia, a-Nangal-Im, Maniyatakan, Krumar-deakan, Indonesia, al-Hal-Yang-angmanai-hay-pah-hang-hang-angi permin-hankakak-wat-an, -lai-lai-lai- salangarakan, tern-an-chara-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a- ...uh-man-Jad-a-tat-Tat-tah-That-tattah-the-tad-tac-tah-tah-tah-t-th-not-tuh-tah-tah-tah-a- huh-the-tah-tat-a-tah-tah-tah-tah-tah-tah-tah-tah-tah-t-the-t-t-t-a-f-tah-t-the-t-t-tah-t-t-t--t-t-t-t-tah-tah- control,-t-t- arch-t-t-tah-tat-t-t-t-h-t-a-not-t-the-t-th-t-t-t-t-t-t-t-t-the-t-t-t-not-y-t-t-
```

### `cahya-whisper-small-id` [int8_float16] (Small — 244M)
- **Speed**: 95.0x RTF (0.511s) | **WER**: 44.44% | **Language**: `id` (prob: 1) | **Words**: 34
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia, hal-hal yang mengenai pemindahan kekuasaan, dan lain-lain diselenggarakan dengan cara saksama, dan dalam tempo yang sesingkat-singkatnya. Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno Ata.
```

### `medium` [int8_float16] (Medium — 769M)
- **Speed**: 61.8x RTF (0.785s) | **WER**: 47.22% | **Language**: `id` (prob: 0.809) | **Words**: 34
```text
proklamasi bangsa Indonesia, dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama. Dan dalam tempuh yang sesingkat-singkatnya, Jakarta 17 Agustus 1945 atas nama bangsa Indonesia, Sukarno Hatta.
```

### `medium.en` [int8_float16] (Medium — 769M)
- **Speed**: 48.6x RTF (0.999s) | **WER**: 105.56% | **Language**: `en` (prob: 1) | **Words**: 44
```text
Rokla Masi, Bangsa, Indonesia. Dungan Inu, Manyatakan, Komerdeka, and Indonesia. Alhau Yangmenga 9, Peminda Han, Kaku Asan, John Line Line, Gisela Ngarakan, Dungan Chara Saktama, Dandalam Temple, Yangtze Tinkap, Tinkapna, Jakarta, Sujubalak Agustus, Seribu, Semilan Ratos, Ampat Pululima, Atak Nama, Bangsa, Indonesia, Sukarno, Atta.
```

### `distil-medium.en` [int8_float16] (Medium — 394M)
- **Speed**: 57.9x RTF (0.838s) | **WER**: 97.22% | **Language**: `en` (prob: 1) | **Words**: 25
```text
Rokkla Masi, Pangsa, Indonesia. Tengan Inu, Mejatakan, Kermardika, and Indonesia. Al-Hao, Yanghmanhman, Permina, Hankakwaraan, Jan Laiin, and Zara Sakthamas, and Zara Sakamah, and Zara Sakamalah,
```

### `cahya-whisper-medium-id` [int8_float16] (Medium — 769M)
- **Speed**: 75.6x RTF (0.642s) | **WER**: 55.56% | **Language**: `id` (prob: 1) | **Words**: 29
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia, hal-hal yang mengenai pemindahan kekuasaan, dan lain-lain diselenggarakan dengan cara saksama. Beratus empat puluh lima atas nama bangsa Indonesia Soekarno Atta.
```

### `large-v1` [int8_float16] (Large — 1550M)
- **Speed**: 47.0x RTF (1.032s) | **WER**: 50.0% | **Language**: `id` (prob: 0.828) | **Words**: 34
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara seksama dan dalam tempoh yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia, Sukarno Hatta.
```

### `large-v2` [int8_float16] (Large — 1550M)
- **Speed**: 49.8x RTF (0.974s) | **WER**: 47.22% | **Language**: `id` (prob: 0.77) | **Words**: 33
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno-Hatta
```

### `large-v3` [int8_float16] (Large — 1550M)
- **Speed**: 48.8x RTF (0.994s) | **WER**: 47.22% | **Language**: `id` (prob: 0.743) | **Words**: 33
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang setingkat-tingkatnya. Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno-Hatta.
```

### `turbo` [int8_float16] (Large — 809M)
- **Speed**: 96.6x RTF (0.502s) | **WER**: 41.67% | **Language**: `id` (prob: 0.995) | **Words**: 34
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno Hatta.
```

### `distil-large-v2` [int8_float16] (Large — 756M)
- **Speed**: 88.0x RTF (0.552s) | **WER**: 94.44% | **Language**: `en` (prob: 0.509) | **Words**: 22
```text
Proclamation Vance Indonesia. With this, it menata Indonesia. All the things that the mingerang the theen the theen thear, and thin thear,
```

### `distil-large-v3` [int8_float16] (Large — 756M)
- **Speed**: 64.8x RTF (0.748s) | **WER**: 86.11% | **Language**: `en` (prob: 0.619) | **Words**: 36
```text
Proclamation of the people Indonesia With this Mnuchamping Kmerdekaan Indonesia All-hall that's on-hannation and other Lain Langeara-kens Deggered with Carta and in Tempo which set-in-cats Gakarta 17 Augustus 2009 944 atas Nama nation Indonesia Sukarno Hata
```

### `tiny` [int8] (Tiny — 39M)
- **Speed**: 113.8x RTF (0.426s) | **WER**: 88.89% | **Language**: `id` (prob: 0.49) | **Words**: 42
```text
Kau kelamati pangsa Indonesia dengan ini menyatakan kemerdilkaan Indonesia, hal hal yang mengenai pemindahan kekuatan, canlai lain, di selengah rakan, telgan cara saksama, canjalam tempu, yang sesingkat kingkatnya. Jakarta, tujubalat Agustus, teribu 70 ratus ampat bululima, atas nama pangsa Indonesia, cukarno hata.
```

### `tiny.en` [int8] (Tiny — 39M)
- **Speed**: 33.5x RTF (1.446s) | **WER**: 100.0% | **Language**: `en` (prob: 1) | **Words**: 0
```text

```

### `cahya-whisper-tiny-id` [int8] (Tiny — 39M)
- **Speed**: 11.6x RTF (4.171s) | **WER**: 130.56% | **Language**: `id` (prob: 1) | **Words**: 54
```text
hau kelamati pangka indonesia dengan ini menyatakan kemerdilkaan indonesia atal,anchext dengan pelebih daran ke kuatan dot enggak lain-lain di saya. Flegareken dengan cara stakama. Tm. yajalam tempo yang setingkap tingkatnya jakarta . Cligi bilat Agustus-giri. fungsi posikan pantikan untuk tepat tepat perpustcutu yang dapat menyelesaixi. Cepat, merasa hanya terhadap telah melancar di atas jadwal.
```

### `base` [int8] (Base — 74M)
- **Speed**: 120.1x RTF (0.404s) | **WER**: 72.22% | **Language**: `id` (prob: 0.965) | **Words**: 36
```text
Jauh kelama di tangsa Indonesia, dengan ini menyatakan kemerdekaan Indonesia. Al-hal yang mengenai pemindahan kekuatan dan lain-lain di selenggarakan dengan cara saksama, dan dalam tempu yang seringkap-tingkapnya, Jakarta, 17 agustus, 1945, atas namapangsa Indonesia, Tukarno, atas.
```

### `base.en` [int8] (Base — 74M)
- **Speed**: 103.2x RTF (0.47s) | **WER**: 119.44% | **Language**: `en` (prob: 1) | **Words**: 48
```text
No calamity, pamsa indonesia, tenan yi, manya takan, kurmertika, and indonesia. Al har yang manna nai permin dahanka kwata an, chan lai nai, isalangara kan, tenan chara saktama, chan dalam tempu, yang satin kapshin kapnya, yakarta, ujubalat argusto, saribu sambirantra to sampakcululima, taktas nama pamsa indonesia, ukarno, taktas.
```

### `small` [int8] (Small — 244M)
- **Speed**: 97.9x RTF (0.495s) | **WER**: 50.0% | **Language**: `id` (prob: 0.627) | **Words**: 34
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempu yang sesingkat-singkatnya. Jakarta 17 Agustus 1945 atas nama bangsa Indonesia, Sukarno Atta.
```

### `small.en` [int8] (Small — 244M)
- **Speed**: 76.7x RTF (0.633s) | **WER**: 100.0% | **Language**: `en` (prob: 1) | **Words**: 43
```text
Rokkala Mahasi, Pangsa, Indonesia, Tengnan Inu, Manyatakan, Karmar Deka, and Indonesia. Hal Hal, Yang Manganai, Praminda Han, Kukwataan, Tan Lai Lai, Yisalanga Arakan, Tengnan Chara Saksama, Tan Dalam Tempo, Yangshasinkap, Tinkapnya, Yakarta, Hujubalas Agustus, Tareebu Similandratos Ampatululima, Atas Nama Pangsa, Indonesia, Tukarno, Ata.
```

### `distil-small.en` [int8] (Small — 166M)
- **Speed**: 78.7x RTF (0.617s) | **WER**: 94.44% | **Language**: `en` (prob: 1) | **Words**: 11
```text
pro-Kla-macy, Pangsa, Indonesia, Rattus Ampatululeima, Apaknama, Pangsa, Indonesia, to Karno, Ata.
```

### `cahya-whisper-small-id` [int8] (Small — 244M)
- **Speed**: 96.3x RTF (0.504s) | **WER**: 44.44% | **Language**: `id` (prob: 1) | **Words**: 34
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia, hal-hal yang mengenai pemindahan kekuasaan, dan lain-lain diselenggarakan dengan cara saksama, dan dalam tempo yang sesingkat-singkatnya. Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno Ata.
```

### `medium` [int8] (Medium — 769M)
- **Speed**: 64.4x RTF (0.753s) | **WER**: 47.22% | **Language**: `id` (prob: 0.809) | **Words**: 34
```text
proklamasi bangsa Indonesia, dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama. Dan dalam tempuh yang sesingkat-singkatnya, Jakarta 17 Agustus 1945 atas nama bangsa Indonesia, Sukarno Hatta.
```

### `medium.en` [int8] (Medium — 769M)
- **Speed**: 48.8x RTF (0.994s) | **WER**: 105.56% | **Language**: `en` (prob: 1) | **Words**: 44
```text
Rokla Masi, Bangsa, Indonesia. Dungan Inu, Manyatakan, Komerdeka, and Indonesia. Alhau Yangmenga 9, Peminda Han, Kaku Asan, John Line Line, Gisela Ngarakan, Dungan Chara Saktama, Dandalam Temple, Yangtze Tinkap, Tinkapna, Jakarta, Sujubalak Agustus, Seribu, Semilan Ratos, Ampat Pululima, Atak Nama, Bangsa, Indonesia, Sukarno, Atta.
```

### `distil-medium.en` [int8] (Medium — 394M)
- **Speed**: 58.0x RTF (0.836s) | **WER**: 97.22% | **Language**: `en` (prob: 1) | **Words**: 25
```text
Rokkla Masi, Pangsa, Indonesia. Tengan Inu, Mejatakan, Kermardika, and Indonesia. Al-Hao, Yanghmanhman, Permina, Hankakwaraan, Jan Laiin, and Zara Sakthamas, and Zara Sakamah, and Zara Sakamalah,
```

### `cahya-whisper-medium-id` [int8] (Medium — 769M)
- **Speed**: 72.0x RTF (0.674s) | **WER**: 55.56% | **Language**: `id` (prob: 1) | **Words**: 29
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia, hal-hal yang mengenai pemindahan kekuasaan, dan lain-lain diselenggarakan dengan cara saksama. Beratus empat puluh lima atas nama bangsa Indonesia Soekarno Atta.
```

### `large-v1` [int8] (Large — 1550M)
- **Speed**: 47.2x RTF (1.029s) | **WER**: 50.0% | **Language**: `id` (prob: 0.828) | **Words**: 34
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara seksama dan dalam tempoh yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia, Sukarno Hatta.
```

### `large-v2` [int8] (Large — 1550M)
- **Speed**: 49.3x RTF (0.984s) | **WER**: 47.22% | **Language**: `id` (prob: 0.77) | **Words**: 33
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno-Hatta
```

### `large-v3` [int8] (Large — 1550M)
- **Speed**: 49.9x RTF (0.973s) | **WER**: 47.22% | **Language**: `id` (prob: 0.743) | **Words**: 33
```text
Proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang setingkat-tingkatnya. Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno-Hatta.
```

### `turbo` [int8] (Large — 809M)
- **Speed**: 99.1x RTF (0.49s) | **WER**: 41.67% | **Language**: `id` (prob: 0.995) | **Words**: 34
```text
proklamasi bangsa Indonesia dengan ini menyatakan kemerdekaan Indonesia. Hal-hal yang mengenai pemindahan kekuasaan dan lain-lain diselenggarakan dengan cara saksama dan dalam tempo yang sesingkat-singkatnya Jakarta 17 Agustus 1945 atas nama bangsa Indonesia Soekarno Hatta.
```

### `distil-large-v2` [int8] (Large — 756M)
- **Speed**: 87.0x RTF (0.558s) | **WER**: 94.44% | **Language**: `en` (prob: 0.509) | **Words**: 22
```text
Proclamation Vance Indonesia. With this, it menata Indonesia. All the things that the mingerang the theen the theen thear, and thin thear,
```

### `distil-large-v3` [int8] (Large — 756M)
- **Speed**: 65.7x RTF (0.739s) | **WER**: 86.11% | **Language**: `en` (prob: 0.619) | **Words**: 36
```text
Proclamation of the people Indonesia With this Mnuchamping Kmerdekaan Indonesia All-hall that's on-hannation and other Lain Langeara-kens Deggered with Carta and in Tempo which set-in-cats Gakarta 17 Augustus 2009 944 atas Nama nation Indonesia Sukarno Hata
```
