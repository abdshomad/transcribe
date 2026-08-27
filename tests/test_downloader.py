from transcribe.downloader import extract_gdrive_id, is_url, _get_cache_path


def test_extract_gdrive_id():
    url1 = "https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing"
    assert extract_gdrive_id(url1) == "1A2B3C4D5E6F7G8H9I0J"

    url2 = "https://drive.google.com/open?id=1234567890abcdef"
    assert extract_gdrive_id(url2) == "1234567890abcdef"

    url3 = "https://drive.google.com/uc?id=xyz987654321"
    assert extract_gdrive_id(url3) == "xyz987654321"

    url4 = "https://example.com/audio.mp3"
    assert extract_gdrive_id(url4) is None


def test_is_url():
    assert is_url("https://drive.google.com/file/d/123/view") is True
    assert is_url("http://example.com/test.wav") is True
    assert is_url("/home/user/audio.mp3") is False
    assert is_url("data/sample/proklamasi.wav") is False


def test_get_cache_path():
    p1 = _get_cache_path("https://drive.google.com/file/d/12345/view")
    assert "gdrive_12345" in p1.name

    p2 = _get_cache_path("https://example.com/test.mp3")
    assert len(p2.stem) >= 16
