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
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .models import (
    CLOUD_MODEL_CATALOG,
    MODEL_CATALOG,
    DiarizedSegment,
    TranscriptSegment,
    TranscriptionResult,
    check_model_cached,
    get_enriched_model_catalog,
)
from .pipeline import AudioTranscriptionPipeline
from .youtube import is_youtube_url, fetch_youtube_transcript
from .downloader import is_gdrive_folder, fetch_gdrive_folder_contents
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

load_dotenv(".secrets")
load_dotenv(".env")
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


@app.get("/api/models")
def get_models():
    """Return catalog of all supported local and cloud ASR models with capabilities and cache state."""
    local_models = [m.model_dump() for m in get_enriched_model_catalog()]
    cloud_models = [m.model_dump() for m in CLOUD_MODEL_CATALOG]
    return JSONResponse(content={
        "local": local_models,
        "cloud": cloud_models,
        "total": len(local_models) + len(cloud_models),
    })


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


def _resolve_resume_context(
    resume_job_id: str,
    default_model: str,
    default_source: str,
) -> Optional[Tuple[float, str, str, str, List[TranscriptSegment]]]:
    """Resolve offset, names, and segments from a prior job for resuming."""
    prior = get_history_item(resume_job_id)
    if not prior:
        return None
    start_offset = float(prior.get("last_processed_time", 0.0))
    effective_name = prior.get("source_name", default_source)
    source_target = prior.get("audio_path") or prior.get("source_name")
    model = prior.get("model", default_model)
    raw_res = prior.get("result", {})
    existing_segs = [TranscriptSegment(**s) for s in raw_res.get("segments", [])]
    return start_offset, effective_name, source_target, model, existing_segs


def _resolve_source_name_path(s_clean: str, upload_dir: Path) -> Optional[str]:
    """Find file on disk or in previous history jobs given a source name."""
    s_name = Path(s_clean).name
    candidate_paths = [
        Path(s_clean),
        upload_dir / s_name,
        Path("data/sample") / s_name,
        Path("data/raw") / s_name,
    ]
    for cp in candidate_paths:
        if cp.exists():
            return str(cp)

    prior_job = find_job_by_source(s_clean)
    if prior_job and prior_job.get("audio_path") and os.path.exists(prior_job["audio_path"]):
        return prior_job["audio_path"]
    if is_url(s_clean):
        return s_clean
    return None


def _handle_youtube_stream(
    source_target: str,
    job_id: str,
    t0: float,
    msg_queue: queue.Queue,
) -> bool:
    """Handle Zero-ASR YouTube subtitle stream. Returns True if handled."""
    if not is_youtube_url(source_target):
        return False

    msg_queue.put({"type": "progress", "data": {"stage": "fetching_youtube_subtitles", "percent": 10.0}})
    yt_res = fetch_youtube_transcript(source_target)
    if not yt_res:
        msg_queue.put({
            "type": "error",
            "error": "Sorry, no existing transcription or subtitles found for this YouTube video."
        })
        return True

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
        msg_queue.put({
            "type": "progress",
            "data": {"stage": "transcribing", "current_time": s.end, "percent": 100.0, "segment": s.model_dump()}
        })
    msg_queue.put({"type": "done", "job_id": job_id, "data": res_data, "processing_time": proc_time})
    return True


def _resolve_upload_or_source(
    file: Optional[UploadFile],
    url: Optional[str],
    source_name: Optional[str],
    upload_dir: Path,
) -> Optional[str]:
    """Resolve file upload, URL string, or named source into a playable audio path."""
    if file and file.filename:
        dest_file = upload_dir / Path(file.filename).name
        with open(dest_file, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        return str(dest_file)
    if url and url.strip():
        return url.strip()
    if source_name and source_name.strip():
        return _resolve_source_name_path(source_name.strip(), upload_dir)
    return None


def _handle_gdrive_folder_stream(
    source_target: str,
    job_id: str,
    model: str,
    language: Optional[str],
    diarize: bool,
    num_speakers: Optional[int],
    compute_type: str,
    beam_size: int,
    vad_filter: bool,
    t0: float,
    msg_queue: queue.Queue,
) -> bool:
    """Handle Google Drive folder batch transcription stream. Returns True if handled."""
    if not is_gdrive_folder(source_target):
        return False

    msg_queue.put({
        "type": "progress",
        "data": {
            "stage": "discovering_gdrive_folder",
            "percent": 5.0,
            "message": "🔍 Scanning Google Drive folder for audio recordings...",
        },
    })
    folder_title, media_files = fetch_gdrive_folder_contents(source_target)
    if not media_files:
        msg_queue.put({
            "type": "error",
            "error": "No audio or video recordings found in this Google Drive folder.",
        })
        return True

    msg_queue.put({
        "type": "progress",
        "data": {
            "stage": "batch_found",
            "percent": 10.0,
            "message": f"📁 Found {len(media_files)} recordings in '{folder_title}'. Starting batch processing...",
        },
    })

    pipeline = AudioTranscriptionPipeline(
        whisper_model_size=model,
        device=DEFAULT_DEVICE,
        compute_type=compute_type,
        enable_diarization=diarize,
    )

    batch_segments = []
    total_files = len(media_files)
    for idx, f in enumerate(media_files, start=1):
        file_pct_base = 10.0 + ((idx - 1) / total_files) * 80.0
        msg_queue.put({
            "type": "progress",
            "data": {
                "stage": "batch_item",
                "percent": file_pct_base,
                "message": f"🎙️ Processing ({idx}/{total_files}): {f['name']}...",
            },
        })
        def item_prog(info: dict) -> None:
            if info.get("stage") == "converting":
                conv_pct = file_pct_base + (info.get("percent", 0.0) / 100.0) * (20.0 / total_files)
                msg_queue.put({
                    "type": "progress",
                    "data": {
                        "stage": "converting",
                        "percent": conv_pct,
                        "message": f"⚙️ Converting ({idx}/{total_files}): {f['name']}... {int(info.get('percent', 0.0))}%",
                    },
                })

        try:
            res = pipeline.process(
                audio_path_or_url=f["url"],
                language=language,
                num_speakers=num_speakers,
                beam_size=beam_size,
                vad_filter=vad_filter,
                on_progress=item_prog,
            )
            item_data = res.model_dump()
            save_history(
                job_id=f"{job_id}_{idx}",
                source_name=f"{folder_title} / {f['name']}",
                model=model,
                result_data=item_data,
                status="completed",
                processing_time=round(time.time() - t0, 2),
                audio_path=f["url"],
            )
            for s in res.segments:
                msg_queue.put({
                    "type": "progress",
                    "data": {
                        "stage": "transcribing",
                        "percent": file_pct_base + (1.0 / total_files) * 80.0,
                        "segment": s.model_dump(),
                    },
                })
            batch_segments.extend(res.segments)
        except Exception as e:
            msg_queue.put({
                "type": "progress",
                "data": {
                    "stage": "item_error",
                    "percent": file_pct_base,
                    "message": f"⚠️ Skipped '{f['name']}': {e}",
                },
            })

    proc_time = round(time.time() - t0, 2)
    final_res = {
        "language": language or "auto",
        "language_probability": 1.0,
        "duration": sum(s.end - s.start for s in batch_segments),
        "segments": [s.model_dump() for s in batch_segments],
        "speakers": sorted(list({s.speaker for s in batch_segments})),
    }
    save_history(
        job_id=job_id,
        source_name=folder_title,
        model=model,
        result_data=final_res,
        status="completed",
        processing_time=proc_time,
        audio_path=source_target,
    )
    msg_queue.put({"type": "done", "job_id": job_id, "data": final_res, "processing_time": proc_time})
    return True


def _run_pipeline_worker(
    job_id: str,
    source_target: str,
    effective_source_name: str,
    model: str,
    language: Optional[str],
    diarize: bool,
    num_speakers: Optional[int],
    start_offset: float,
    existing_segs: List[TranscriptSegment],
    t0: float,
    msg_queue: queue.Queue,
    done_flag: threading.Event,
    compute_type: str = "default",
    beam_size: int = 5,
    vad_filter: bool = True,
    use_itn: bool = True,
    chunk_length_s: float = 30.0,
    target_lang: Optional[str] = None,
) -> None:
    """Run transcription worker in background thread with checkpointing."""
    try:
        if _handle_youtube_stream(source_target, job_id, t0, msg_queue):
            return

        if _handle_gdrive_folder_stream(
            source_target=source_target,
            job_id=job_id,
            model=model,
            language=language,
            diarize=diarize,
            num_speakers=num_speakers,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            t0=t0,
            msg_queue=msg_queue,
        ):
            return

        if not check_model_cached(model):
            msg_queue.put({
                "type": "progress",
                "data": {
                    "stage": "downloading",
                    "percent": 0.0,
                    "message": f"📥 Downloading model '{model}' to local server cache...",
                },
            })

        pipeline = AudioTranscriptionPipeline(
            whisper_model_size=model,
            device=DEFAULT_DEVICE,
            compute_type=compute_type,
            enable_diarization=diarize,
        )
        current_res_holder = {
            "segments": [s.model_dump() for s in existing_segs],
            "language": "auto",
            "duration": 0.0,
            "speakers": [],
        }

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

        lang = language if language and language.strip() else None
        res = pipeline.process(
            audio_path_or_url=source_target,
            language=lang,
            num_speakers=num_speakers,
            start_offset=start_offset,
            existing_segments=existing_segs,
            beam_size=beam_size,
            vad_filter=vad_filter,
            use_itn=use_itn,
            chunk_length_s=chunk_length_s,
            target_lang=target_lang,
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


async def _sse_stream_generator(
    job_id: str,
    source_target: str,
    effective_source_name: str,
    model: str,
    language: Optional[str],
    diarize: bool,
    num_speakers: Optional[int],
    start_offset: float,
    existing_segs: List[TranscriptSegment],
    t0: float,
    compute_type: str = "default",
    beam_size: int = 5,
    vad_filter: bool = True,
    use_itn: bool = True,
    chunk_length_s: float = 30.0,
    target_lang: Optional[str] = None,
):
    """Yield SSE formatted messages as background transcription worker progresses."""
    msg_queue: queue.Queue = queue.Queue()
    done_flag = threading.Event()

    for seg in existing_segs:
        msg_queue.put({
            "type": "progress",
            "data": {"stage": "transcribing", "current_time": seg.end, "percent": 0.0, "segment": seg.model_dump()}
        })

    thread = threading.Thread(
        target=_run_pipeline_worker,
        args=(
            job_id,
            source_target,
            effective_source_name,
            model,
            language,
            diarize,
            num_speakers,
            start_offset,
            existing_segs,
            t0,
            msg_queue,
            done_flag,
            compute_type,
            beam_size,
            vad_filter,
            use_itn,
            chunk_length_s,
            target_lang,
        ),
        daemon=True,
    )
    thread.start()

    while not done_flag.is_set() or not msg_queue.empty():
        try:
            item = msg_queue.get_nowait()
            yield f"data: {json.dumps(item)}\n\n"
        except queue.Empty:
            await asyncio.sleep(0.08)


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
    compute_type: str = Form("default"),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    use_itn: bool = Form(True),
    chunk_length_s: float = Form(30.0),
    target_lang: Optional[str] = Form(None),
):
    """Real-time SSE stream protected by token verification."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    start_offset = 0.0
    existing_segs: List[TranscriptSegment] = []
    effective_source_name = file.filename if (file and file.filename) else (url or source_name or "audio")
    source_target = None
    t0 = time.time()

    if resume_job_id and not force:
        job_id = resume_job_id
        res_ctx = _resolve_resume_context(resume_job_id, model, effective_source_name)
        if res_ctx:
            start_offset, effective_source_name, source_target, model, existing_segs = res_ctx
    else:
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    if not source_target:
        source_target = _resolve_upload_or_source(file, url, source_name, upload_dir)

    if not source_target and not resume_job_id:
        raise HTTPException(status_code=400, detail="Audio input, valid source_name, or resume_job_id must be provided")

    return StreamingResponse(
        _sse_stream_generator(
            job_id,
            source_target or "",
            effective_source_name,
            model,
            language,
            diarize,
            num_speakers,
            start_offset,
            existing_segs,
            t0,
            compute_type,
            beam_size,
            vad_filter,
            use_itn,
            chunk_length_s,
            target_lang,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def start():
    """Start uvicorn server."""
    print(f"Starting Transcribe server on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    start()
