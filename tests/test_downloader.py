import pytest
from transcribe.downloader import extract_gdrive_id, is_url, is_gdrive_folder, _get_cache_path


def test_extract_gdrive_id():
    url1 = "https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing"
    assert extract_gdrive_id(url1) == "1A2B3C4D5E6F7G8H9I0J"

    url2 = "https://drive.google.com/open?id=1234567890abcdef"
    assert extract_gdrive_id(url2) == "1234567890abcdef"

    url3 = "https://drive.google.com/uc?id=xyz987654321"
    assert extract_gdrive_id(url3) == "xyz987654321"

    url4 = "https://example.com/audio.mp3"
    assert extract_gdrive_id(url4) is None

    folder_url = "https://drive.google.com/drive/folders/1H-zm-0SccZoLuUZ-h2paJerbnjnlYaV4?usp=sharing"
    assert is_gdrive_folder(folder_url) is True
    with pytest.raises(ValueError, match="folder link"):
        extract_gdrive_id(folder_url)

    doc_url = "https://docs.google.com/document/d/15RepS3Q75JgdGC9n8c2fTZVtKWKCyyPuhxdcF9cUDqo/edit"
    with pytest.raises(ValueError, match="Google Docs/Sheets text document"):
        extract_gdrive_id(doc_url)


def test_is_url():
    assert is_url("https://drive.google.com/file/d/123/view") is True
    assert is_url("http://example.com/test.wav") is True
    assert is_url("/home/user/audio.mp3") is False
    assert is_url("data/sample/proklamasi.wav") is False


def test_is_media_candidate():
    from transcribe.downloader import is_media_candidate

    # Recognized audio extensions
    assert is_media_candidate("audio.mp3") is True
    assert is_media_candidate("speech.wav") is True
    assert is_media_candidate("recording.m4a") is True
    assert is_media_candidate("video.mp4") is True
    assert is_media_candidate("track.flac") is True

    # Extensionless Google Meet recordings
    assert is_media_candidate("Meeting MVP Akreditasi 2026 - Recording 1") is True
    assert is_media_candidate("Weekly Sync Audio") is True
    assert is_media_candidate("Customer Voice Call") is True

    # Excluded Google Docs / non-media files
    assert is_media_candidate("MVP Akreditasi 2026 - Notes by Gemini") is False
    assert is_media_candidate("Document.pdf") is False
    assert is_media_candidate("Budget.xlsx") is False
    assert is_media_candidate("Presentation.pptx") is False
    assert is_media_candidate("notes.txt") is False


def test_extract_gdrive_folder_id():
    from transcribe.downloader import extract_gdrive_folder_id, is_gdrive_folder

    url1 = "https://drive.google.com/drive/folders/1H-zm-0SccZoLuUZ-h2paJerbnjnlYaV4?usp=sharing"
    assert extract_gdrive_folder_id(url1) == "1H-zm-0SccZoLuUZ-h2paJerbnjnlYaV4"
    assert is_gdrive_folder(url1) is True

    url2 = "https://drive.google.com/drive/u/0/folders/12345abcdef67890"
    assert extract_gdrive_folder_id(url2) == "12345abcdef67890"
    assert is_gdrive_folder(url2) is True

    file_url = "https://drive.google.com/file/d/12345/view"
    assert extract_gdrive_folder_id(file_url) is None
    assert is_gdrive_folder(file_url) is False


def test_download_headers_resume(tmp_path):
    from transcribe.downloader import _prepare_download_headers

    empty_file = tmp_path / "empty.tmp"
    headers_empty, bytes_empty = _prepare_download_headers(empty_file)
    assert "Range" not in headers_empty
    assert bytes_empty == 0

    partial_file = tmp_path / "partial.tmp"
    partial_file.write_bytes(b"1234567890")
    headers_partial, bytes_partial = _prepare_download_headers(partial_file)
    assert headers_partial.get("Range") == "bytes=10-"
    assert bytes_partial == 10

