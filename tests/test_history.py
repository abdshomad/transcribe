from transcribe.history import (
    save_history,
    update_history_item,
    list_history,
    list_sources,
    compare_runs,
    get_history_item,
    delete_history_item,
)


def test_history_crud_and_patch():
    job_id = "test_job_patch_456"
    result_data = {
        "language": "id",
        "duration": 12.5,
        "speakers": ["Speaker 1"],
        "segments": [
            {"id": 0, "speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "Hello initial test."}
        ],
    }

    item = save_history(job_id, "sample.wav", "tiny", result_data, processing_time=1.2)
    assert item.id == job_id
    assert item.processing_time == 1.2

    # Test update / rename speaker
    result_data["speakers"] = ["Dr. Smith"]
    result_data["segments"][0]["speaker"] = "Dr. Smith"
    updated = update_history_item(job_id, result_data)
    assert updated is True

    fetched = get_history_item(job_id)
    assert fetched["result"]["speakers"] == ["Dr. Smith"]
    assert fetched["result"]["segments"][0]["speaker"] == "Dr. Smith"

    deleted = delete_history_item(job_id)
    assert deleted is True


def test_multi_model_storage_and_comparison():
    src = "lecture_recording.mp3"
    job_tiny = "job_test_tiny_101"
    job_large = "job_test_large_102"

    res_tiny = {
        "language": "en",
        "duration": 60.0,
        "speakers": ["Speaker 1"],
        "segments": [
            {"id": 0, "speaker": "Speaker 1", "start": 0.0, "end": 4.0, "text": "welcome to the machine learning lecture today"}
        ],
    }
    res_large = {
        "language": "en",
        "duration": 60.0,
        "speakers": ["Speaker 1", "Speaker 2"],
        "segments": [
            {"id": 0, "speaker": "Speaker 1", "start": 0.0, "end": 4.0, "text": "Welcome to the deep machine learning lecture today."}
        ],
    }

    save_history(job_tiny, src, "tiny", res_tiny, processing_time=2.5)
    save_history(job_large, src, "large-v3", res_large, processing_time=12.0)

    # Verify both runs are stored
    item_tiny = get_history_item(job_tiny)
    item_large = get_history_item(job_large)
    assert item_tiny is not None
    assert item_large is not None
    assert item_tiny["model"] == "tiny"
    assert item_large["model"] == "large-v3"

    # Verify sources grouping
    sources = list_sources()
    matched_src = next((s for s in sources if s["source_name"] == src), None)
    assert matched_src is not None
    assert len(matched_src["runs"]) >= 2
    assert "tiny" in matched_src["models"]
    assert "large-v3" in matched_src["models"]

    # Verify comparison
    comp = compare_runs(job_tiny, job_large)
    assert comp is not None
    assert comp["similarity_score"] > 80.0
    assert comp["run_a"]["model"] == "tiny"
    assert comp["run_b"]["model"] == "large-v3"
    assert comp["run_a"]["processing_time"] == 2.5
    assert comp["run_b"]["processing_time"] == 12.0
    assert len(comp["run_a"]["diff_words"]) > 0
    assert len(comp["run_b"]["diff_words"]) > 0

    # Cleanup
    delete_history_item(job_tiny)
    delete_history_item(job_large)


def test_find_checkpoint():
    from transcribe.history import find_checkpoint

    src = "checkpoint_test_recording.mp3"
    job_cp = "job_cp_test_999"
    res_partial = {
        "language": "en",
        "duration": 120.0,
        "speakers": ["SPEAKER_00"],
        "segments": [
            {"id": 0, "speaker": "SPEAKER_00", "start": 0.0, "end": 10.0, "text": "Part 1"}
        ],
    }
    save_history(
        job_id=job_cp,
        source_name=src,
        model="base",
        result_data=res_partial,
        status="in_progress",
        last_processed_time=10.0,
        processing_time=1.5,
    )

    cp = find_checkpoint(src, "base")
    assert cp is not None
    assert cp["id"] == job_cp
    assert cp["last_processed_time"] == 10.0
    assert len(cp["segments"]) == 1

    delete_history_item(job_cp)


def test_is_sub_part_recording_and_source_filtering():
    from transcribe.history import is_sub_part_recording

    parents = {"Meeting MVP Sistem Akreditasi Rumah Sakit 2026 - 2026-08-28 0807 WIB"}
    assert is_sub_part_recording("MVP Akreditasi Rumah Sakit 2026 - 2026_08_28 08:07 WIB - Recording 1.m4a", parents) is True
    assert is_sub_part_recording("MVP Akreditasi Rumah Sakit 2026 - 2026_08_28 08:07 WIB - Recording 2", parents) is True
    assert is_sub_part_recording("Meeting MVP Sistem Akreditasi Rumah Sakit 2026 - 2026-08-28 0807 WIB", parents) is False
    assert is_sub_part_recording("proklamasi.wav", parents) is False

