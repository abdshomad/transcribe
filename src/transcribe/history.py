"""Transcription history, progressive checkpoints, multi-model storage, and comparison engine."""

import difflib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

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
        ]:
            try:
                conn.execute(f"ALTER TABLE transcriptions ADD COLUMN {col} {col_type} DEFAULT {dflt}")
            except sqlite3.OperationalError:
                pass

        # Create indexes for high-speed lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at ON transcriptions (created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcriptions_source ON transcriptions (source_name)")

        # Seed DEMO token if table is empty
        cur = conn.execute("SELECT COUNT(*) FROM api_tokens")
        if cur.fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO api_tokens (token, label, created_at, is_active) VALUES ('DEMO', 'Default Demo Token', ?, 1)",
                (time.time(),),
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
    speakers_count = len(result_data.get("speakers", []))
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


def list_history(limit: int = 100) -> List[Dict[str, Any]]:
    """List recent transcriptions across all models."""
    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, source_name, model, language, duration, speakers_count, created_at, snippet, status, last_processed_time, processing_time, audio_path
            FROM transcriptions ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def list_sources() -> List[Dict[str, Any]]:
    """List distinct audio sources grouped with their model runs for comparison."""
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

    return list(grouped.values())


def get_history_item(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full transcription result by ID."""
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM transcriptions WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        res["result"] = json.loads(res["result_json"])
        return res


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


def compare_runs(job_id_a: str, job_id_b: str) -> Optional[Dict[str, Any]]:
    """Compare two transcription runs, generating word diffs, alignment, and performance metrics."""
    run_a = get_history_item(job_id_a)
    run_b = get_history_item(job_id_b)
    if not run_a or not run_b:
        return None

    res_a = run_a.get("result", {})
    res_b = run_b.get("result", {})

    segs_a = res_a.get("segments", [])
    segs_b = res_b.get("segments", [])

    text_a = " ".join(s.get("text", "").strip() for s in segs_a)
    text_b = " ".join(s.get("text", "").strip() for s in segs_b)

    words_a = text_a.split()
    words_b = text_b.split()

    def norm(t: str) -> str:
        return t.strip().lower().strip(".,!?;:\"'()[]{}")

    norm_a = [norm(w) for w in words_a]
    norm_b = [norm(w) for w in words_b]

    matcher = difflib.SequenceMatcher(None, norm_a, norm_b)
    similarity = matcher.ratio() * 100.0

    diff_a: List[Dict[str, str]] = []
    diff_b: List[Dict[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for w in words_a[i1:i2]:
                diff_a.append({"word": w, "status": "equal"})
            for w in words_b[j1:j2]:
                diff_b.append({"word": w, "status": "equal"})
        elif tag == "delete":
            for w in words_a[i1:i2]:
                diff_a.append({"word": w, "status": "deleted"})
        elif tag == "insert":
            for w in words_b[j1:j2]:
                diff_b.append({"word": w, "status": "inserted"})
        elif tag == "replace":
            for w in words_a[i1:i2]:
                diff_a.append({"word": w, "status": "replaced"})
            for w in words_b[j1:j2]:
                diff_b.append({"word": w, "status": "replaced"})

    dur_a = float(run_a.get("duration", 0.0))
    proc_a = float(run_a.get("processing_time", 0.0))
    speed_a = round(dur_a / proc_a, 2) if proc_a > 0 else 0.0

    dur_b = float(run_b.get("duration", 0.0))
    proc_b = float(run_b.get("processing_time", 0.0))
    speed_b = round(dur_b / proc_b, 2) if proc_b > 0 else 0.0

    return {
        "source_name": run_a.get("source_name") or run_b.get("source_name"),
        "similarity_score": round(similarity, 1),
        "run_a": {
            "id": run_a["id"],
            "model": run_a["model"],
            "language": run_a["language"],
            "duration": dur_a,
            "processing_time": proc_a,
            "speedup": speed_a,
            "word_count": len(words_a),
            "speakers_count": run_a.get("speakers_count", 0),
            "segments": segs_a,
            "diff_words": diff_a,
            "created_at": run_a["created_at"],
        },
        "run_b": {
            "id": run_b["id"],
            "model": run_b["model"],
            "language": run_b["language"],
            "duration": dur_b,
            "processing_time": proc_b,
            "speedup": speed_b,
            "word_count": len(words_b),
            "speakers_count": run_b.get("speakers_count", 0),
            "segments": segs_b,
            "diff_words": diff_b,
            "created_at": run_b["created_at"],
        },
    }
