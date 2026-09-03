"""SQLite persistence for tasks — survives server restarts."""
import json
import sqlite3
import threading

from .config import DB_PATH

_lock = threading.Lock()

_FIELDS = ("id", "filename", "source", "status", "stage",
          "summary_status", "analytics_status", "error", "created_at", "updated_at")


def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                filename TEXT,
                source TEXT,
                status TEXT,
                stage TEXT,
                summary_status TEXT,
                analytics_status TEXT,
                error TEXT,
                created_at TEXT,
                updated_at TEXT,
                result_json TEXT
            )
        """)
        # Admin portal: runtime-editable settings (key -> JSON value).
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Issued X-API-Keys (only the hash is stored; prefix is for display).
        c.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                label TEXT,
                key_hash TEXT,
                prefix TEXT,
                created_at TEXT,
                revoked_at TEXT
            )
        """)
        # Webhook delivery log + retry queue (one row per task-event to the global URL).
        c.execute("""
            CREATE TABLE IF NOT EXISTS webhook_events (
                id TEXT PRIMARY KEY,
                task_id TEXT,
                event_type TEXT,
                target_url TEXT,
                status TEXT,
                attempts INTEGER DEFAULT 0,
                last_error TEXT,
                payload_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)


def save(job: dict):
    """Upsert a task row from an in-memory job dict (result stored as JSON)."""
    row = {k: job.get(k) for k in _FIELDS}
    result = job.get("result")
    result_json = json.dumps(result, ensure_ascii=False) if result else None
    with _lock, sqlite3.connect(DB_PATH) as c:
        c.execute("""
            INSERT INTO tasks (id, filename, source, status, stage, summary_status,
                               analytics_status, error, created_at, updated_at, result_json)
            VALUES (:id, :filename, :source, :status, :stage, :summary_status,
                    :analytics_status, :error, :created_at, :updated_at, :result_json)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, stage=excluded.stage,
                summary_status=excluded.summary_status,
                analytics_status=excluded.analytics_status,
                error=excluded.error, updated_at=excluded.updated_at,
                result_json=COALESCE(excluded.result_json, tasks.result_json)
        """, {**row, "result_json": result_json})


def _row_to_job(r: sqlite3.Row) -> dict:
    job = {k: r[k] for k in _FIELDS}
    job["result"] = json.loads(r["result_json"]) if r["result_json"] else None
    return job


def load_all() -> list:
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        return [_row_to_job(r) for r in c.execute("SELECT * FROM tasks ORDER BY created_at")]


def delete_task(tid: str):
    with _lock, sqlite3.connect(DB_PATH) as c:
        c.execute("DELETE FROM tasks WHERE id=?", (tid,))
        c.execute("DELETE FROM webhook_events WHERE task_id=?", (tid,))


# ---------------------------------------------------------------- settings (key -> JSON)
def get_setting(key: str, default=None):
    with sqlite3.connect(DB_PATH) as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default


def set_setting(key: str, value):
    with _lock, sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, json.dumps(value, ensure_ascii=False)))


# ---------------------------------------------------------------- api keys
def add_api_key(kid: str, label: str, key_hash: str, prefix: str, created_at: str):
    with _lock, sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO api_keys(id, label, key_hash, prefix, created_at, revoked_at) "
                  "VALUES(?, ?, ?, ?, ?, NULL)", (kid, label, key_hash, prefix, created_at))


def list_api_keys() -> list:
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            "SELECT id, label, prefix, created_at, revoked_at FROM api_keys ORDER BY created_at DESC")]


def active_key_hashes() -> set:
    with sqlite3.connect(DB_PATH) as c:
        return {r[0] for r in c.execute(
            "SELECT key_hash FROM api_keys WHERE revoked_at IS NULL")}


def revoke_api_key(kid: str, when: str) -> bool:
    with _lock, sqlite3.connect(DB_PATH) as c:
        cur = c.execute("UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                        (when, kid))
        return cur.rowcount > 0


# ---------------------------------------------------------------- webhook events
def add_webhook_event(eid, task_id, event_type, target_url, payload, now):
    with _lock, sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO webhook_events(id, task_id, event_type, target_url, status, "
                  "attempts, last_error, payload_json, created_at, updated_at) "
                  "VALUES(?, ?, ?, ?, 'pending', 0, NULL, ?, ?, ?)",
                  (eid, task_id, event_type, target_url,
                   json.dumps(payload, ensure_ascii=False), now, now))


def update_webhook_event(eid, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock, sqlite3.connect(DB_PATH) as c:
        c.execute(f"UPDATE webhook_events SET {cols} WHERE id=?", (*fields.values(), eid))


def deliverable_events(max_attempts: int) -> list:
    """Pending events, or failed ones that still have retry attempts left."""
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            "SELECT * FROM webhook_events WHERE status='pending' "
            "OR (status='failed' AND attempts < ?) ORDER BY created_at", (max_attempts,))]


def list_webhook_events(limit: int = 200, task_id: str | None = None) -> list:
    q = "SELECT id, task_id, event_type, target_url, status, attempts, last_error, " \
        "created_at, updated_at FROM webhook_events"
    args: tuple = ()
    if task_id:
        q += " WHERE task_id=?"
        args = (task_id,)
    q += " ORDER BY created_at DESC LIMIT ?"
    args += (limit,)
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, args)]


def get_webhook_event(eid: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        r = c.execute("SELECT * FROM webhook_events WHERE id=?", (eid,)).fetchone()
    return dict(r) if r else None
