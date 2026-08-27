#!/usr/bin/env python3
"""
Playwright automated E2E test runner and screenshot capture suite.
Saves 5-step visual screenshots to:
screenshots/e2e/{model_group}/{folder_name}/{step_num}-{step_name}.jpg
"""

import os
import sys
import time
import socket
import threading
import uvicorn
from pathlib import Path
from playwright.sync_api import sync_playwright

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from transcribe.server import app

SCREENSHOT_DIR = WORKSPACE_ROOT / "screenshots" / "e2e"
AUDIO_SAMPLES = {
    "en": WORKSPACE_ROOT / "data" / "sample" / "jfk.wav",
    "id": WORKSPACE_ROOT / "data" / "sample" / "proklamasi.wav",
}


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_server(port: int, stop_event: threading.Event):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    
    def check_stop():
        while not stop_event.is_set():
            time.sleep(0.5)
        server.should_exit = True

    t = threading.Thread(target=check_stop, daemon=True)
    t.start()
    server.run()


TEST_MODELS = [
    # Sub-Plan 1: Faster-Whisper
    {"group": "whisper", "folder": "whisper-tiny-default", "family": "Faster-Whisper", "variant": "tiny", "compute": "default", "sample": "en"},
    {"group": "whisper", "folder": "whisper-base-default", "family": "Faster-Whisper", "variant": "base", "compute": "default", "sample": "en"},
    {"group": "whisper", "folder": "whisper-small-default", "family": "Faster-Whisper", "variant": "small", "compute": "default", "sample": "en"},
    {"group": "whisper", "folder": "whisper-turbo-default", "family": "Faster-Whisper", "variant": "turbo", "compute": "default", "sample": "id"},
    # Sub-Plan 2: SenseVoice
    {"group": "sensevoice", "folder": "sensevoice-small-fp16", "family": "Alibaba SenseVoice", "variant": "sensevoice-small", "compute": "default", "sample": "en"},
    # Sub-Plan 3: Moonshine
    {"group": "moonshine", "folder": "moonshine-tiny-onnx", "family": "UsefulSensors Moonshine", "variant": "moonshine-tiny", "compute": "default", "sample": "en"},
    {"group": "moonshine", "folder": "moonshine-base-onnx", "family": "UsefulSensors Moonshine", "variant": "moonshine-base", "compute": "default", "sample": "en"},
    # Sub-Plan 4: Meta MMS
    {"group": "mms", "folder": "mms-1b-all", "family": "Meta MMS", "variant": "meta-omnilingual-asr", "compute": "default", "sample": "id"},
    # Sub-Plan 5: Indonesian CTC
    {"group": "wav2vec2", "folder": "wav2vec2-regional-id-jv-su", "family": "Indonesian CTC", "variant": "indonesian-wav2vec2-regional", "compute": "default", "sample": "id"},
    {"group": "wav2vec2", "folder": "wav2vec2-large-xlsr-id", "family": "Indonesian CTC", "variant": "indonesian-wav2vec2-large-xlsr", "compute": "default", "sample": "id"},
]


def execute_e2e_screenshots():
    port = find_free_port()
    stop_event = threading.Event()
    server_thread = threading.Thread(target=run_server, args=(port, stop_event), daemon=True)
    server_thread.start()
    time.sleep(2.0)

    url = f"http://127.0.0.1:{port}"
    print(f"🚀 Started test server on {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.add_init_script("localStorage.setItem('appToken', 'DEMO');")

        for idx, cfg in enumerate(TEST_MODELS, 1):
            out_dir = SCREENSHOT_DIR / cfg["group"] / cfg["folder"]
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_file = AUDIO_SAMPLES.get(cfg["sample"], AUDIO_SAMPLES["en"])

            print(f"[{idx}/{len(TEST_MODELS)}] Testing {cfg['family']} ➔ {cfg['variant']} ({cfg['folder']})...")
            try:
                # Step 1: Initial Clean Load
                page.goto(url)
                page.wait_for_selector("#family-select")
                time.sleep(0.6)
                page.screenshot(path=str(out_dir / "01-initial-state.jpg"), quality=85, type="jpeg")

                # Step 2: Model Configured
                page.select_option("#family-select", cfg["family"])
                time.sleep(0.3)
                page.select_option("#variant-select", cfg["variant"])
                time.sleep(0.3)
                if cfg.get("compute") and cfg["compute"] != "default":
                    page.select_option("#compute-type-select", cfg["compute"])
                    time.sleep(0.2)
                page.screenshot(path=str(out_dir / "02-model-configured.jpg"), quality=85, type="jpeg")

                # Step 3: File Uploaded
                page.set_input_files("#file-input", str(sample_file))
                page.wait_for_selector("#file-info:not(.hidden)")
                time.sleep(0.4)
                page.screenshot(path=str(out_dir / "03-file-uploaded.jpg"), quality=85, type="jpeg")

                # Step 4: Streaming Progress
                page.click("#btn-transcribe")
                time.sleep(1.5)
                page.screenshot(path=str(out_dir / "04-streaming-progress.jpg"), quality=85, type="jpeg")

                # Step 5: Completed Results (Wait for completion text)
                page.wait_for_selector("#progress-stage-text:has-text('Completed')", timeout=180000)
                time.sleep(1.0)
                page.screenshot(path=str(out_dir / "05-completed-results.jpg"), quality=85, type="jpeg")
                print(f"  ✅ Saved 5 screenshots to {out_dir.relative_to(WORKSPACE_ROOT)}")

            except Exception as ex:
                print(f"  ❌ Error testing {cfg['variant']}: {ex}")
                try:
                    page.screenshot(path=str(out_dir / "error-snapshot.jpg"), quality=85, type="jpeg")
                except Exception:
                    pass
                existing_issues = sorted((WORKSPACE_ROOT / "issues").glob("[0-9][0-9][0-9]-*.md"))
                next_num = f"{len(existing_issues) + 1:03d}"
                issue_file = WORKSPACE_ROOT / "issues" / f"{next_num}-{cfg['group']}-{cfg['variant']}-e2e-error.md"
                with open(issue_file, "w", encoding="utf-8") as f:
                    f.write(f"# Issue {next_num}: E2E Error - {cfg['family']} {cfg['variant']}\n\n"
                            f"**Error**: `{ex}`\n\n"
                            f"**Config**: `{cfg}`\n\n"
                            f"**Status**: OPEN\n")

        browser.close()

    stop_event.set()
    print("✨ E2E Visual Verification Suite completed successfully!")


if __name__ == "__main__":
    execute_e2e_screenshots()
