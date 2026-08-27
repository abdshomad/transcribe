"""Common evaluation metrics for ASR benchmarks (WER, CER, RTF, text normalization)."""

import re
from typing import Dict, Any, Optional


def normalize_text(text: str) -> str:
    """Normalize text for WER/CER comparison: lowercases, strips punctuation and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_wer(ref: str, hyp: str) -> float:
    """
    Calculate Word Error Rate (WER) percentage using Levenshtein distance.
    Returns: float between 0.0 and 100.0 (or higher if insertion count exceeds ref length).
    """
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 100.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)

    return round((d[len(ref_words)][len(hyp_words)] / len(ref_words)) * 100, 2)


def calculate_cer(ref: str, hyp: str) -> float:
    """
    Calculate Character Error Rate (CER) percentage using Levenshtein distance.
    Returns: float between 0.0 and 100.0.
    """
    ref_chars = list(normalize_text(ref).replace(" ", ""))
    hyp_chars = list(normalize_text(hyp).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 100.0

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)

    return round((d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)) * 100, 2)


def calculate_rtf(audio_duration_s: float, elapsed_s: float) -> float:
    """
    Calculate Real-Time Factor (RTF) speed multiplier.
    e.g. 48.5s audio processed in 0.5s = 97.0x speed.
    """
    if elapsed_s <= 0:
        return 0.0
    return round(audio_duration_s / elapsed_s, 1)
