"""Transcription history, progressive checkpoints, multi-model storage, and comparison engine."""

import difflib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(".secrets")
load_dotenv(".env")

DB_PATH = Path("data/history.db")


class HistoryItem(BaseModel):
    """Transcription history entry."""

    id: str
    source_name: str
    model: str
    language: str
    duration: float
    speakers_count: int
    created_at: float
    snippet: str
    status: str = "completed"
    last_processed_time: float = 0.0
    processing_time: float = 0.0
    audio_path: Optional[str] = None
    mom_markdown: Optional[str] = None
    refined_text: Optional[str] = None
    result_json: str


_DB_INITIALIZED = False


def _init_db(conn: sqlite3.Connection) -> None:
    """Run schema creation, migrations, and index setups once."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id TEXT PRIMARY KEY,
                source_name TEXT,
                model TEXT,
                language TEXT,
                duration REAL,
                speakers_count INTEGER,
                created_at REAL,
                snippet TEXT,
                status TEXT DEFAULT 'completed',
                last_processed_time REAL DEFAULT 0.0,
                processing_time REAL DEFAULT 0.0,
                audio_path TEXT,
                mom_markdown TEXT,
                refined_text TEXT,
                result_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_tokens (
                token TEXT PRIMARY KEY,
                label TEXT,
                created_at REAL,
                is_active INTEGER DEFAULT 1
            )
        """)
        for col, col_type, dflt in [
            ("status", "TEXT", "'completed'"),
            ("last_processed_time", "REAL", "0.0"),
            ("processing_time", "REAL", "0.0"),
            ("audio_path", "TEXT", "NULL"),
            ("mom_markdown", "TEXT", "NULL"),
            ("refined_text", "TEXT", "NULL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE transcriptions ADD COLUMN {col} {col_type} DEFAULT {dflt}")
            except sqlite3.OperationalError:
                pass

        # Create indexes for high-speed lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at ON transcriptions (created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcriptions_source ON transcriptions (source_name)")

        # Seed configured env token
        env_token = os.getenv("TOKEN", "DEMO").strip()
        conn.execute(
            "INSERT OR IGNORE INTO api_tokens (token, label, created_at, is_active) VALUES (?, 'Active Token', ?, 1)",
            (env_token, time.time()),
        )
        conn.execute(
            "UPDATE api_tokens SET is_active = 1 WHERE token = ?",
            (env_token,),
        )


def _get_db() -> sqlite3.Connection:
    """Return an optimized SQLite connection."""
    global _DB_INITIALIZED
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    if not _DB_INITIALIZED:
        _init_db(conn)
        _DB_INITIALIZED = True
    return conn


def is_valid_token(token: Optional[str]) -> bool:
    """Validate token against active records in SQLite."""
    if not token:
        return False
    with _get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM api_tokens WHERE token = ? AND is_active = 1",
            (token.strip(),),
        ).fetchone()
        return row is not None


def find_job_by_source_and_model(source_name: str, model: str) -> Optional[Dict[str, Any]]:
    """Find existing job ID by source name and model."""
    with _get_db() as conn:
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE source_name = ? AND model = ? ORDER BY created_at DESC LIMIT 1",
            (source_name, model),
        ).fetchone()
        return dict(row) if row else None


def find_job_by_source(source_name: str) -> Optional[Dict[str, Any]]:
    """Find latest job ID by source name."""
    with _get_db() as conn:
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE source_name = ? ORDER BY created_at DESC LIMIT 1",
            (source_name,),
        ).fetchone()
        return dict(row) if row else None


def find_checkpoint(source_name: str, model: str) -> Optional[Dict[str, Any]]:
    """Find recoverable in-progress checkpoint for source and model."""
    with _get_db() as conn:
        row = conn.execute(
            """
            SELECT * FROM transcriptions 
            WHERE source_name = ? AND model = ? AND status != 'completed' AND last_processed_time > 0
            ORDER BY created_at DESC LIMIT 1
            """,
            (source_name, model),
        ).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res_data = json.loads(res.get("result_json", "{}"))
            res["segments"] = res_data.get("segments", [])
        except Exception:
            res["segments"] = []
        return res


def save_history(
    job_id: str,
    source_name: str,
    model: str,
    result_data: Dict[str, Any],
    status: str = "completed",
    last_processed_time: Optional[float] = None,
    processing_time: float = 0.0,
    audio_path: Optional[str] = None,
) -> HistoryItem:
    """Save or update transcription record, preserving distinct runs for comparisons."""
    segments = result_data.get("segments", [])
    snippet = " ".join(s.get("text", "") for s in segments[:3]).strip()[:180]
    duration = float(result_data.get("duration", 0.0))
    language = str(result_data.get("language", "unknown"))
    speakers_count = len(result_data.get("speakers", []))
    created_at = time.time()
    last_proc = last_processed_time if last_processed_time is not None else duration
    result_json = json.dumps(result_data, ensure_ascii=False)

    item = HistoryItem(
        id=job_id,
        source_name=source_name,
        model=model,
        language=language,
        duration=duration,
        speakers_count=speakers_count,
        created_at=created_at,
        snippet=snippet,
        status=status,
        last_processed_time=last_proc,
        processing_time=processing_time,
        audio_path=audio_path,
        result_json=result_json,
    )

    with _get_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO transcriptions
            (id, source_name, model, language, duration, speakers_count, created_at, snippet, status, last_processed_time, processing_time, audio_path, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.source_name,
                item.model,
                item.language,
                item.duration,
                item.speakers_count,
                item.created_at,
                item.snippet,
                item.status,
                item.last_processed_time,
                item.processing_time,
                item.audio_path,
                item.result_json,
            ),
        )
    return item


def checkpoint_segment(
    job_id: str,
    source_name: str,
    model: str,
    segment: Dict[str, Any],
    duration: float,
    current_result: Dict[str, Any],
    audio_path: Optional[str] = None,
    processing_time: float = 0.0,
) -> None:
    """Progressively commit segment checkpoint to SQLite."""
    save_history(
        job_id=job_id,
        source_name=source_name,
        model=model,
        result_data=current_result,
        status="in_progress",
        last_processed_time=float(segment.get("end", 0.0)),
        processing_time=processing_time,
        audio_path=audio_path,
    )


def update_history_item(job_id: str, result_data: Dict[str, Any]) -> bool:
    """Update existing transcription result, snippet, and speakers."""
    segments = result_data.get("segments", [])
    snippet = " ".join(s.get("text", "") for s in segments[:3]).strip()[:180]
    distinct_speakers = list(dict.fromkeys(s.get("speaker") for s in segments if s.get("speaker")))
    if not result_data.get("speakers"):
        result_data["speakers"] = distinct_speakers
    speakers_count = len(distinct_speakers) if distinct_speakers else len(result_data.get("speakers", []))
    result_json = json.dumps(result_data, ensure_ascii=False)

    with _get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE transcriptions
            SET snippet = ?, speakers_count = ?, result_json = ?
            WHERE id = ?
            """,
            (snippet, speakers_count, result_json, job_id),
        )
        return cursor.rowcount > 0


def is_sub_part_recording(name: str, parent_sources: set[str]) -> bool:
    """Check if name is a sub-recording fragment when a parent batch exists."""
    is_part_pattern = bool(re.search(r'(?: - | / |_| )?(?:Recording|Part|Track)\s*\d+', name, re.IGNORECASE))
    if not is_part_pattern:
        return False
    for p in parent_sources:
        if p == name:
            continue
        p_tokens = set(re.findall(r'[a-zA-Z0-9]+', p.lower())) - {'meeting', 'session', 'wib'}
        c_tokens = set(re.findall(r'[a-zA-Z0-9]+', name.lower()))
        if len(p_tokens.intersection(c_tokens)) >= 3:
            return True
    return False


def list_history(limit: int = 100) -> List[Dict[str, Any]]:
    """List recent transcriptions across all models, omitting child parts if parent exists."""
    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, source_name, model, language, duration, speakers_count, created_at, snippet, status, last_processed_time, processing_time, audio_path
            FROM transcriptions ORDER BY created_at DESC
            """
        ).fetchall()
        items = [dict(row) for row in rows]

    all_sources = {item["source_name"] for item in items}
    filtered = [
        item for item in items
        if not is_sub_part_recording(item["source_name"], all_sources)
    ]
    return filtered[:limit]


def list_sources() -> List[Dict[str, Any]]:
    """List distinct audio sources grouped with their model runs for comparison, filtering child fragments."""
    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, source_name, model, language, duration, speakers_count, created_at, snippet, status, last_processed_time, processing_time, audio_path
            FROM transcriptions
            ORDER BY created_at DESC
            """
        ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        item = dict(r)
        src = item["source_name"]
        if src not in grouped:
            grouped[src] = {
                "source_name": src,
                "latest_created_at": item["created_at"],
                "duration": item["duration"],
                "audio_path": item["audio_path"],
                "models": [],
                "runs": [],
            }
        grouped[src]["runs"].append(item)
        if item["model"] not in grouped[src]["models"]:
            grouped[src]["models"].append(item["model"])

    all_sources = set(grouped.keys())
    filtered = [
        src_obj for src, src_obj in grouped.items()
        if not is_sub_part_recording(src, all_sources)
    ]
    return filtered


def get_history_item(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full transcription result by ID."""
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM transcriptions WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        res["result"] = json.loads(res["result_json"])
        return res


def save_history_mom(job_id: str, mom_text: str) -> bool:
    """Save generated Minutes of Meeting markdown for a history entry."""
    with _get_db() as conn:
        cursor = conn.execute("UPDATE transcriptions SET mom_markdown = ? WHERE id = ?", (mom_text, job_id))
        return cursor.rowcount > 0


def get_history_mom(job_id: str) -> Optional[str]:
    """Retrieve generated Minutes of Meeting markdown for a history entry."""
    with _get_db() as conn:
        row = conn.execute("SELECT mom_markdown FROM transcriptions WHERE id = ?", (job_id,)).fetchone()
        return row[0] if row and row[0] else None


def save_history_refined(job_id: str, refined_text: str) -> bool:
    """Save polished/refined transcript text for a history entry."""
    with _get_db() as conn:
        cursor = conn.execute("UPDATE transcriptions SET refined_text = ? WHERE id = ?", (refined_text, job_id))
        return cursor.rowcount > 0


def get_history_refined(job_id: str) -> Optional[str]:
    """Retrieve polished/refined transcript text for a history entry."""
    with _get_db() as conn:
        row = conn.execute("SELECT refined_text FROM transcriptions WHERE id = ?", (job_id,)).fetchone()
        return row[0] if row and row[0] else None


def delete_history_item(job_id: str) -> bool:
    """Delete a transcription history record."""
    with _get_db() as conn:
        cursor = conn.execute("DELETE FROM transcriptions WHERE id = ?", (job_id,))
        return cursor.rowcount > 0


def clear_all_history() -> int:
    """Delete all transcription history records."""
    with _get_db() as conn:
        cursor = conn.execute("DELETE FROM transcriptions")
        return cursor.rowcount


def _normalize_diff_word(w: str) -> str:
    """Normalize word token for alignment comparison."""
    return w.strip().lower().strip(".,!?;:\"'()[]{}")


def _compute_word_diffs(
    words_a: List[str],
    words_b: List[str],
) -> Tuple[float, List[Dict[str, str]], List[Dict[str, str]]]:
    """Compute token difference opcodes and similarity ratio."""
    norm_a = [_normalize_diff_word(w) for w in words_a]
    norm_b = [_normalize_diff_word(w) for w in words_b]
    matcher = difflib.SequenceMatcher(None, norm_a, norm_b)
    similarity = matcher.ratio() * 100.0

    diff_a: List[Dict[str, str]] = []
    diff_b: List[Dict[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            diff_a.extend({"word": w, "status": "equal"} for w in words_a[i1:i2])
            diff_b.extend({"word": w, "status": "equal"} for w in words_b[j1:j2])
        elif tag == "delete":
            diff_a.extend({"word": w, "status": "deleted"} for w in words_a[i1:i2])
        elif tag == "insert":
            diff_b.extend({"word": w, "status": "inserted"} for w in words_b[j1:j2])
        elif tag == "replace":
            diff_a.extend({"word": w, "status": "replaced"} for w in words_a[i1:i2])
            diff_b.extend({"word": w, "status": "replaced"} for w in words_b[j1:j2])

    return round(similarity, 1), diff_a, diff_b


def _build_run_summary(
    run_dict: Dict[str, Any],
    words: List[str],
    diff_list: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Extract metrics and metadata for one comparison run."""
    dur = float(run_dict.get("duration", 0.0))
    proc = float(run_dict.get("processing_time", 0.0))
    speed = round(dur / proc, 2) if proc > 0 else 0.0
    segs = run_dict.get("result", {}).get("segments", [])

    return {
        "id": run_dict["id"],
        "model": run_dict["model"],
        "language": run_dict["language"],
        "duration": dur,
        "processing_time": proc,
        "speedup": speed,
        "word_count": len(words),
        "speakers_count": run_dict.get("speakers_count", 0),
        "segments": segs,
        "full_text": " ".join(s.get("text", "").strip() for s in segs),
        "diff_words": diff_list,
    }


def compare_runs(job_id_a: str, job_id_b: str) -> Optional[Dict[str, Any]]:
    """Compare two transcription runs, generating word diffs, alignment, and performance metrics."""
    run_a = get_history_item(job_id_a)
    run_b = get_history_item(job_id_b)
    if not run_a or not run_b:
        return None

    words_a = " ".join(s.get("text", "").strip() for s in run_a.get("result", {}).get("segments", [])).split()
    words_b = " ".join(s.get("text", "").strip() for s in run_b.get("result", {}).get("segments", [])).split()

    similarity, diff_a, diff_b = _compute_word_diffs(words_a, words_b)

    return {
        "source_name": run_a.get("source_name") or run_b.get("source_name"),
        "similarity_score": similarity,
        "run_a": _build_run_summary(run_a, words_a, diff_a),
        "run_b": _build_run_summary(run_b, words_b, diff_b),
        "delta": {
            "duration_diff": round(float(run_b.get("duration", 0)) - float(run_a.get("duration", 0)), 2),
            "speedup_diff": round((_build_run_summary(run_b, words_b, diff_b)["speedup"]) - (_build_run_summary(run_a, words_a, diff_a)["speedup"]), 2),
            "word_count_diff": len(words_b) - len(words_a),
            "speakers_diff": int(run_b.get("speakers_count", 0)) - int(run_a.get("speakers_count", 0)),
        },
    }
