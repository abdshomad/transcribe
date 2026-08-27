"""FastAPI REST & Web server with database token security and real-time streaming."""

import asyncio
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .downloader import is_url
from .models import TranscriptSegment, DiarizedSegment, TranscriptionResult
from .pipeline import AudioTranscriptionPipeline
from .youtube import is_youtube_url, fetch_youtube_transcript
from .history import (
    is_valid_token,
    save_history,
    checkpoint_segment,
    update_history_item,
    find_job_by_source,
    list_history,
    list_sources,
    compare_runs,
    get_history_item,
    delete_history_item,
    clear_all_history,
)
from .web import HTML_PAGE

load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "4013"))
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")
DEFAULT_DEVICE = os.getenv("DEVICE", "auto")

app = FastAPI(
    title="Transcribe API",
    description="Speech-to-text API with Faster-Whisper, Real-Time Diarization, Multi-Model Storage and Run Comparison",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_token(
    request: Request,
    token: Optional[str] = Query(None),
    x_api_token: Optional[str] = Header(None, alias="X-API-Token"),
    authorization: Optional[str] = Header(None),
) -> str:
    """Verify access token against database."""
    auth_tok = None
    if token and is_valid_token(token):
        auth_tok = token
    elif x_api_token and is_valid_token(x_api_token):
        auth_tok = x_api_token
    elif authorization and authorization.startswith("Bearer "):
        bearer_val = authorization.split("Bearer ", 1)[1].strip()
        if is_valid_token(bearer_val):
            auth_tok = bearer_val

    if not auth_tok:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid API Token Required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_tok


@app.get("/", response_class=HTMLResponse)
def get_index():
    """Serve single-page frontend application."""
    return HTML_PAGE


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "app": "Transcribe", "host": HOST, "port": PORT}


@app.get("/api/auth/verify")
def check_token(tok: str = Depends(verify_token)):
    """Validate token validity."""
    return {"status": "valid", "token": tok}


@app.get("/api/history", dependencies=[Depends(verify_token)])
def get_history():
    """List past transcription jobs."""
    return JSONResponse(content=list_history())


@app.get("/api/sources", dependencies=[Depends(verify_token)])
def get_sources():
    """List distinct audio sources with their model runs for comparison."""
    return JSONResponse(content=list_sources())


@app.get("/api/compare", dependencies=[Depends(verify_token)])
def get_comparison(job_a: str = Query(...), job_b: str = Query(...)):
    """Compare two transcription runs by model and size."""
    comp = compare_runs(job_a, job_b)
    if not comp:
        raise HTTPException(status_code=404, detail="One or both transcription runs not found")
    return JSONResponse(content=comp)


@app.delete("/api/history", dependencies=[Depends(verify_token)])
def clear_history():
    """Clear all past transcription history."""
    count = clear_all_history()
    return {"status": "cleared", "deleted_count": count}


@app.get("/api/history/{job_id}", dependencies=[Depends(verify_token)])
def get_history_entry(job_id: str):
    """Get single transcription details and segments."""
    entry = get_history_item(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=entry)


@app.patch("/api/history/{job_id}", dependencies=[Depends(verify_token)])
def update_history_entry(job_id: str, payload: Dict[str, Any] = Body(...)):
    """Update transcription result."""
    if not update_history_item(job_id, payload):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "updated", "id": job_id}


@app.delete("/api/history/{job_id}", dependencies=[Depends(verify_token)])
def delete_history_entry(job_id: str):
    """Delete a transcription record."""
    if not delete_history_item(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "id": job_id}


@app.post("/api/transcribe-stream", dependencies=[Depends(verify_token)])
async def transcribe_audio_stream(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    source_name: Optional[str] = Form(None),
    model: str = Form(DEFAULT_MODEL),
    language: Optional[str] = Form(None),
    diarize: bool = Form(True),
    num_speakers: Optional[int] = Form(None),
    resume_job_id: Optional[str] = Form(None),
    force: bool = Form(False),
):
    """Real-time SSE stream protected by token verification."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    start_offset = 0.0
    existing_segs = []
    effective_source_name = file.filename if (file and file.filename) else (url or source_name or "audio")
    source_target = None
    t0 = time.time()

    if resume_job_id and not force:
        job_id = resume_job_id
        prior = get_history_item(job_id)
        if prior:
            start_offset = float(prior.get("last_processed_time", 0.0))
            effective_source_name = prior.get("source_name", effective_source_name)
            source_target = prior.get("audio_path") or prior.get("source_name")
            model = prior.get("model", model)
            raw_res = prior.get("result", {})
            existing_segs = [TranscriptSegment(**s) for s in raw_res.get("segments", [])]
    else:
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    if not source_target:
        if file and file.filename:
            safe_name = Path(file.filename).name
            dest_file = upload_dir / safe_name
            with open(dest_file, "wb") as f_out:
                shutil.copyfileobj(file.file, f_out)
            source_target = str(dest_file)
        elif url and url.strip():
            source_target = url.strip()
        elif source_name and source_name.strip():
            s_clean = source_name.strip()
            s_name = Path(s_clean).name
            if os.path.exists(s_clean):
                source_target = s_clean
            elif (upload_dir / s_name).exists():
                source_target = str(upload_dir / s_name)
            elif (Path("data/sample") / s_name).exists():
                source_target = str(Path("data/sample") / s_name)
            elif (Path("data/raw") / s_name).exists():
                source_target = str(Path("data/raw") / s_name)
            else:
                prior_job = find_job_by_source(s_clean)
                if prior_job and prior_job.get("audio_path") and os.path.exists(prior_job["audio_path"]):
                    source_target = prior_job["audio_path"]
                elif is_url(s_clean):
                    source_target = s_clean

    if not source_target and not resume_job_id:
        raise HTTPException(status_code=400, detail="Audio input, valid source_name, or resume_job_id must be provided")

    async def event_generator():
        msg_queue = queue.Queue()
        done_flag = threading.Event()
        pipeline = AudioTranscriptionPipeline(
            whisper_model_size=model,
            device=DEFAULT_DEVICE,
            enable_diarization=diarize,
        )
        current_res_holder = {
            "segments": [s.model_dump() for s in existing_segs],
            "language": "auto",
            "duration": 0.0,
            "speakers": [],
        }

        for seg in existing_segs:
            msg_queue.put({"type": "progress", "data": {"stage": "transcribing", "current_time": seg.end, "percent": 0.0, "segment": seg.model_dump()}})

        def on_prog(info):
            if info.get("stage") == "transcribing" and "segment" in info:
                current_res_holder["segments"].append(info["segment"])
                checkpoint_segment(
                    job_id=job_id,
                    source_name=effective_source_name,
                    model=model,
                    segment=info["segment"],
                    duration=info.get("duration", 0.0),
                    current_result=current_res_holder,
                    audio_path=source_target,
                    processing_time=round(time.time() - t0, 2),
                )
            msg_queue.put({"type": "progress", "data": info})

        def run_worker():
            try:
                # Handle YouTube URL Zero-ASR Fast Path
                if is_youtube_url(source_target):
                    msg_queue.put({"type": "progress", "data": {"stage": "fetching_youtube_subtitles", "percent": 10.0}})
                    yt_res = fetch_youtube_transcript(source_target)
                    if yt_res:
                        proc_time = round(time.time() - t0, 2)
                        diarized_segs = [
                            DiarizedSegment(
                                id=s.id,
                                speaker="SPEAKER_00",
                                start=s.start,
                                end=s.end,
                                text=s.text,
                                words=s.words
                            )
                            for s in yt_res["segments"]
                        ]
                        trans_result = TranscriptionResult(
                            language=yt_res["language"],
                            language_probability=1.0,
                            duration=yt_res["duration"],
                            segments=diarized_segs,
                            speakers=["SPEAKER_00"]
                        )
                        res_data = trans_result.model_dump()
                        save_history(
                            job_id=job_id,
                            source_name=yt_res["title"],
                            model="youtube-captions",
                            result_data=res_data,
                            status="completed",
                            processing_time=proc_time,
                            audio_path=source_target,
                        )
                        for s in diarized_segs:
                            msg_queue.put({"type": "progress", "data": {"stage": "transcribing", "current_time": s.end, "percent": 100.0, "segment": s.model_dump()}})
                        msg_queue.put({"type": "done", "job_id": job_id, "data": res_data, "processing_time": proc_time})
                        return
                    else:
                        msg_queue.put({
                            "type": "error",
                            "error": "Sorry, no existing transcription or subtitles found for this YouTube video."
                        })
                        return

                lang = language if language and language.strip() else None
                res = pipeline.process(
                    audio_path_or_url=source_target,
                    language=lang,
                    num_speakers=num_speakers,
                    start_offset=start_offset,
                    existing_segments=existing_segs,
                    on_progress=on_prog,
                )
                res_data = res.model_dump()
                proc_time = round(time.time() - t0, 2)
                save_history(
                    job_id=job_id,
                    source_name=effective_source_name,
                    model=model,
                    result_data=res_data,
                    status="completed",
                    processing_time=proc_time,
                    audio_path=source_target,
                )
                msg_queue.put({"type": "done", "job_id": job_id, "data": res_data, "processing_time": proc_time})
            except Exception as e:
                msg_queue.put({"type": "error", "error": str(e)})
            finally:
                done_flag.set()

        thread = threading.Thread(target=run_worker, daemon=True)
        thread.start()

        while not done_flag.is_set() or not msg_queue.empty():
            try:
                item = msg_queue.get_nowait()
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.08)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def start():
    """Start uvicorn server."""
    print(f"Starting Transcribe server on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    start()
