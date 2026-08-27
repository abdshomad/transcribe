"""Unit tests for YouTube URL detection and subtitle extraction."""

from transcribe.youtube import is_youtube_url, parse_vtt_content, parse_vtt_timestamp, fetch_youtube_transcript


def test_is_youtube_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=rsCGWaO-rbI") is True
    assert is_youtube_url("http://youtu.be/rsCGWaO-rbI") is True
    assert is_youtube_url("https://youtube.com/shorts/rsCGWaO-rbI") is True
    assert is_youtube_url("https://example.com/audio.mp3") is False
    assert is_youtube_url("data/sample/proklamasi.wav") is False


def test_parse_vtt_timestamp():
    assert parse_vtt_timestamp("00:01:23.456") == 83.456
    assert parse_vtt_timestamp("01:23.456") == 83.456
    assert parse_vtt_timestamp("00:00:05.000") == 5.0


def test_parse_vtt_content():
    sample_vtt = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:04.500
<v Speaker>Hello world this is a test caption.</v>

00:00:05.000 --> 00:00:08.000
Second subtitle line here.
"""
    segments, full_text = parse_vtt_content(sample_vtt)
    assert len(segments) == 2
    assert segments[0].start == 1.0
    assert segments[0].end == 4.5
    assert "Hello world this is a test caption." in segments[0].text
    assert "Second subtitle line here." in full_text


def test_fetch_youtube_transcript_invalid():
    res = fetch_youtube_transcript("https://www.youtube.com/watch?v=invalid_id_999999")
    assert res is None
