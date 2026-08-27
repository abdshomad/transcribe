"""Unit tests for ASR evaluation metrics (WER, CER, RTF, normalization)."""

from transcribe.metrics import normalize_text, calculate_wer, calculate_cer, calculate_rtf


def test_normalize_text():
    raw = "  Hello, World! This IS a Test... \n"
    expected = "hello world this is a test"
    assert normalize_text(raw) == expected


def test_calculate_wer_exact_match():
    ref = "Kami bangsa Indonesia dengan ini menjatakan kemerdekaan Indonesia"
    hyp = "Kami bangsa Indonesia dengan ini menjatakan kemerdekaan Indonesia"
    assert calculate_wer(ref, hyp) == 0.0


def test_calculate_wer_with_errors():
    ref = "ask what you can do for your country"
    hyp = "ask what you could do for our country"
    wer = calculate_wer(ref, hyp)
    assert 0.0 < wer < 50.0


def test_calculate_wer_empty():
    assert calculate_wer("", "") == 0.0
    assert calculate_wer("hello world", "") == 100.0


def test_calculate_cer():
    ref = "Indonesia"
    hyp = "Indonisia"
    cer = calculate_cer(ref, hyp)
    assert 0.0 < cer < 20.0


def test_calculate_rtf():
    # 48.5s processed in 0.5s = 97.0x speed
    assert calculate_rtf(48.5, 0.5) == 97.0
    assert calculate_rtf(10.0, 0.0) == 0.0
