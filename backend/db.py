"""PostgreSQL persistence for the ASR portal (dedicated instance on :5433).

Design (see the admin data-model):
- `recordings`   — one row per job/task + the latest full result blob.
- `runs`         — EVERY transcribe/summary/analytics execution, versioned and
                   never overwritten, with the exact `params` used and the full
                   `result` snapshot. This is the audit trail for accuracy
                   verification and model/prompt comparison over time.
- normalized `turns`, `speaker_stats`, `summaries`, `analytics`,
  `scorecard_items` — extracted from each run so results are SQL-queryable.
- `settings` / `api_keys` / `webhook_events` — runtime config + ops.

The public function names (save/load_all/delete_task/get_setting/... /webhook)
match the old SQLite module so callers are unchanged; `record_run` is new.
"""
import threading
import uuid
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from .config import DATABASE_URL

_lock = threading.Lock()

_FIELDS = ("id", "filename", "source", "status", "stage",
           "summary_status", "analytics_status", "error", "created_at", "updated_at")


def _conn():
    return psycopg.connect(DATABASE_URL, autocommit=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _role_of(roles: dict, spk):
    if roles.get("agent") == spk:
        return "agent"
    if roles.get("customer") == spk:
        return "customer"
    return None


# ---------------------------------------------------------------- schema
_DDL = [
    """CREATE TABLE IF NOT EXISTS recordings (
        id TEXT PRIMARY KEY, filename TEXT, source TEXT, status TEXT, stage TEXT,
        summary_status TEXT, analytics_status TEXT, error TEXT,
        created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ, result_json JSONB,
        request_params JSONB)""",
    "ALTER TABLE recordings ADD COLUMN IF NOT EXISTS request_params JSONB",
    "CREATE INDEX IF NOT EXISTS idx_rec_status ON recordings(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_rec_sumq ON recordings(summary_status)",
    "CREATE INDEX IF NOT EXISTS idx_rec_anaq ON recordings(analytics_status)",
    """CREATE TABLE IF NOT EXISTS runs (
        run_id UUID PRIMARY KEY,
        recording_id TEXT REFERENCES recordings(id) ON DELETE CASCADE,
        run_type TEXT, status TEXT, params JSONB, result JSONB,
        processing_seconds DOUBLE PRECISION, created_at TIMESTAMPTZ DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS idx_runs_rec ON runs(recording_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_type ON runs(run_type, created_at DESC)",
    """CREATE TABLE IF NOT EXISTS turns (
        id BIGSERIAL PRIMARY KEY,
        run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
        recording_id TEXT, idx INT, speaker TEXT, role TEXT,
        start_s DOUBLE PRECISION, end_s DOUBLE PRECISION, text TEXT, text_translated TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_turns_run ON turns(run_id)",
    """CREATE TABLE IF NOT EXISTS speaker_stats (
        id BIGSERIAL PRIMARY KEY,
        run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
        recording_id TEXT, speaker TEXT, role TEXT,
        talk_seconds DOUBLE PRECISION, talk_share_pct DOUBLE PRECISION)""",
    """CREATE TABLE IF NOT EXISTS summaries (
        run_id UUID PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
        recording_id TEXT, summary TEXT, outcome TEXT, customer_sentiment JSONB,
        agent_tone TEXT, key_points JSONB, action_items JSONB,
        agent_label TEXT, customer_label TEXT)""",
    """CREATE TABLE IF NOT EXISTS analytics (
        run_id UUID PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
        recording_id TEXT, overall_score INT, compliance_fail BOOLEAN)""",
    """CREATE TABLE IF NOT EXISTS scorecard_items (
        id BIGSERIAL PRIMARY KEY,
        run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
        recording_id TEXT, category TEXT, checkpoint TEXT, score DOUBLE PRECISION,
        verdict TEXT, evidence TEXT, suggestion TEXT)""",
    "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value JSONB)",
    # Multiple admins can log in with their own email + password (pbkdf2 hash).
    """CREATE TABLE IF NOT EXISTS admin_users (
        id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT,
        password_hash TEXT NOT NULL, active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMPTZ DEFAULT now(), last_login TIMESTAMPTZ)""",
    """CREATE TABLE IF NOT EXISTS api_keys (
        id TEXT PRIMARY KEY, label TEXT, key_hash TEXT, prefix TEXT,
        created_at TIMESTAMPTZ, revoked_at TIMESTAMPTZ)""",
    """CREATE TABLE IF NOT EXISTS webhook_events (
        id TEXT PRIMARY KEY, task_id TEXT, event_type TEXT, target_url TEXT, status TEXT,
        attempts INT DEFAULT 0, last_error TEXT, payload_json JSONB,
        created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)""",
    # Stable keys + normalized verdict flags for rollups (survive checkpoint renames).
    "ALTER TABLE scorecard_items ADD COLUMN IF NOT EXISTS category_id TEXT",
    "ALTER TABLE scorecard_items ADD COLUMN IF NOT EXISTS checkpoint_id TEXT",
    "ALTER TABLE scorecard_items ADD COLUMN IF NOT EXISTS met INT",
    "ALTER TABLE scorecard_items ADD COLUMN IF NOT EXISTS applicable BOOLEAN",
    "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS scorecard_version INT",
    "CREATE INDEX IF NOT EXISTS idx_sci_cpid ON scorecard_items(checkpoint_id)",
]


def init_db():
    with _conn() as c, c.cursor() as cur:
        for stmt in _DDL:
            cur.execute(stmt)


# ---------------------------------------------------------------- recordings (task state)
def save(job: dict):
    """Upsert a recording row from an in-memory job dict (latest result as JSONB)."""
    row = {k: job.get(k) for k in _FIELDS}
    result = job.get("result")
    with _conn() as c, c.cursor() as cur:
        cur.execute("""
            INSERT INTO recordings (id, filename, source, status, stage, summary_status,
                                    analytics_status, error, created_at, updated_at, result_json)
            VALUES (%(id)s, %(filename)s, %(source)s, %(status)s, %(stage)s, %(summary_status)s,
                    %(analytics_status)s, %(error)s, %(created_at)s, %(updated_at)s, %(result_json)s)
            ON CONFLICT (id) DO UPDATE SET
                status=EXCLUDED.status, stage=EXCLUDED.stage,
                summary_status=EXCLUDED.summary_status,
                analytics_status=EXCLUDED.analytics_status,
                error=EXCLUDED.error, updated_at=EXCLUDED.updated_at,
                result_json=COALESCE(EXCLUDED.result_json, recordings.result_json)
        """, {**row, "result_json": Json(result) if result else None})


def _row_to_job(r: dict) -> dict:
    job = {k: r.get(k) for k in _FIELDS}
    for tk in ("created_at", "updated_at"):
        if job.get(tk) is not None:
            job[tk] = job[tk].isoformat()
    job["result"] = r.get("result_json")
    return job


def load_all() -> list:
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM recordings ORDER BY created_at")
        return [_row_to_job(r) for r in cur.fetchall()]


def delete_task(tid: str):
    with _conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM recordings WHERE id=%s", (tid,))   # cascades runs/turns/...
        cur.execute("DELETE FROM webhook_events WHERE task_id=%s", (tid,))


# ---------------------------------------------------------------- durable queue
def enqueue_recording(jid: str, filename: str, source: str, request_params: dict):
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO recordings (id, filename, source, status, request_params, "
            "created_at, updated_at) VALUES (%s,%s,%s,'queued',%s,%s,%s)",
            (jid, filename, source, Json(request_params or {}), _now(), _now()))


def load_one(tid: str) -> dict | None:
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM recordings WHERE id=%s", (tid,))
        r = cur.fetchone()
    return _row_to_job(r) if r else None


def list_recordings() -> list:
    """Metadata only (no heavy result_json), newest first — for the dashboard list."""
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, filename, source, status, stage, summary_status, "
                    "analytics_status, error, created_at, updated_at FROM recordings "
                    "ORDER BY created_at DESC")
        rows = cur.fetchall()
    for r in rows:
        for tk in ("created_at", "updated_at"):
            if r.get(tk) is not None:
                r[tk] = r[tk].isoformat()
    return rows


def claim_transcribe() -> dict | None:
    """Atomically claim the next queued recording for a worker (concurrency-safe)."""
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            UPDATE recordings SET status='running', stage='Ingesting', updated_at=%s
            WHERE id = (SELECT id FROM recordings WHERE status='queued'
                        ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
            RETURNING id, source, request_params
        """, (_now(),))
        return cur.fetchone()


def claim_stage(stage: str) -> dict | None:
    """Atomically claim a queued on-demand stage (summary|analytics) on a done recording."""
    col = "summary_status" if stage == "summary" else "analytics_status"
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            UPDATE recordings SET {col}='running', updated_at=%s
            WHERE id = (SELECT id FROM recordings WHERE {col}='queued' AND status='done'
                        ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
            RETURNING id, result_json, request_params
        """, (_now(),))
        return cur.fetchone()


def set_recording(tid: str, **fields):
    """Patch a recording's columns (pass result=<dict> to store the result blob)."""
    if "result" in fields:
        fields["result_json"] = Json(fields.pop("result"))
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k}=%s" for k in fields)
    with _conn() as c, c.cursor() as cur:
        cur.execute(f"UPDATE recordings SET {cols} WHERE id=%s", (*fields.values(), tid))


def queue_counts() -> dict:
    with _conn() as c, c.cursor() as cur:
        cur.execute("SELECT status, count(*) FROM recordings GROUP BY status")
        by = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute("SELECT count(*) FROM recordings WHERE summary_status IN ('queued','running') "
                    "OR analytics_status IN ('queued','running')")
        stages = cur.fetchone()[0]
    return {"queued": by.get("queued", 0), "running": by.get("running", 0),
            "done": by.get("done", 0), "error": by.get("error", 0), "active_stages": stages}


def requeue_running():
    """Startup recovery: work that was mid-flight didn't finish → back on the queue."""
    with _conn() as c, c.cursor() as cur:
        cur.execute("UPDATE recordings SET status='queued', stage=NULL WHERE status='running'")
        cur.execute("UPDATE recordings SET summary_status='queued' WHERE summary_status='running'")
        cur.execute("UPDATE recordings SET analytics_status='queued' WHERE analytics_status='running'")


# ---------------------------------------------------------------- versioned runs (audit trail)
def record_run(recording_id: str, run_type: str, params: dict | None, result: dict) -> str:
    """Store one execution of a stage as an immutable, versioned run + normalized rows.

    run_type: 'transcribe' | 'summary' | 'analytics'. Never updates existing rows.
    """
    run_id = uuid.uuid4()
    proc = (result or {}).get("elapsed_seconds") or (result or {}).get("processing_seconds")
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (run_id, recording_id, run_type, status, params, result, "
            "processing_seconds, created_at) VALUES (%s,%s,%s,'done',%s,%s,%s,%s)",
            (run_id, recording_id, run_type, Json(params or {}), Json(result or {}), proc, _now()))
        if run_type == "transcribe":
            _insert_turns(cur, run_id, recording_id, result or {})
        elif run_type == "summary":
            _insert_summary(cur, run_id, recording_id, result or {})
        elif run_type == "analytics":
            _insert_analytics(cur, run_id, recording_id, result or {})
    return str(run_id)


def _insert_turns(cur, run_id, rec, result):
    roles = result.get("roles") or {}
    for i, t in enumerate(result.get("turns") or []):
        cur.execute(
            "INSERT INTO turns (run_id, recording_id, idx, speaker, role, start_s, end_s, "
            "text, text_translated) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (run_id, rec, i, t.get("speaker"), _role_of(roles, t.get("speaker")),
             t.get("start"), t.get("end"), t.get("text"), t.get("text_translated")))
    for spk, v in ((result.get("stats") or {}).get("per_speaker") or {}).items():
        cur.execute(
            "INSERT INTO speaker_stats (run_id, recording_id, speaker, role, talk_seconds, "
            "talk_share_pct) VALUES (%s,%s,%s,%s,%s,%s)",
            (run_id, rec, spk, _role_of(roles, spk), v.get("talk_seconds"), v.get("talk_share_pct")))


def _insert_summary(cur, run_id, rec, result):
    s = result.get("summary") or {}
    roles = s.get("roles") or result.get("roles") or {}
    cur.execute(
        "INSERT INTO summaries (run_id, recording_id, summary, outcome, customer_sentiment, "
        "agent_tone, key_points, action_items, agent_label, customer_label) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (run_id, rec, s.get("summary"), s.get("outcome"), Json(s.get("customer_sentiment")),
         s.get("agent_tone"), Json(s.get("key_points") or []), Json(s.get("action_items") or []),
         roles.get("agent"), roles.get("customer")))


def _insert_analytics(cur, run_id, rec, result):
    a = result.get("analytics") or {}
    cur.execute(
        "INSERT INTO analytics (run_id, recording_id, overall_score, compliance_fail, "
        "scorecard_version) VALUES (%s,%s,%s,%s,%s)",
        (run_id, rec, a.get("overall_score"), a.get("compliance_fail"), a.get("scorecard_version")))
    for it in a.get("scorecard") or []:
        cur.execute(
            "INSERT INTO scorecard_items (run_id, recording_id, category, category_id, "
            "checkpoint, checkpoint_id, score, verdict, met, applicable, evidence, suggestion) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (run_id, rec, it.get("category"), it.get("category_id"), it.get("checkpoint"),
             it.get("checkpoint_id"), it.get("score"), it.get("verdict"), it.get("met"),
             it.get("applicable"), it.get("evidence"), it.get("suggestion")))


def runs_for(recording_id: str) -> list:
    """All runs for a recording, newest first, with a headline metric per run."""
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT r.run_id, r.run_type, r.status, r.params, r.processing_seconds, r.created_at,
                   a.overall_score, a.compliance_fail, s.outcome,
                   (SELECT count(*) FROM turns t WHERE t.run_id = r.run_id) AS turn_count
            FROM runs r
            LEFT JOIN analytics a ON a.run_id = r.run_id
            LEFT JOIN summaries s ON s.run_id = r.run_id
            WHERE r.recording_id = %s
            ORDER BY r.created_at DESC
        """, (recording_id,))
        rows = cur.fetchall()
    for r in rows:
        r["run_id"] = str(r["run_id"])
        if r.get("created_at") is not None:
            r["created_at"] = r["created_at"].isoformat()
    return rows


def get_run(run_id: str) -> dict | None:
    """A single run with its full params + stored result snapshot."""
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT run_id, recording_id, run_type, status, params, result, "
                    "processing_seconds, created_at FROM runs WHERE run_id = %s::uuid", (run_id,))
        r = cur.fetchone()
    if not r:
        return None
    r["run_id"] = str(r["run_id"])
    if r.get("created_at") is not None:
        r["created_at"] = r["created_at"].isoformat()
    return r


# ---------------------------------------------------------------- settings (key -> JSON)
def get_setting(key: str, default=None):
    with _conn() as c, c.cursor() as cur:
        cur.execute("SELECT value FROM settings WHERE key=%s", (key,))
        r = cur.fetchone()
    return r[0] if r else default


def set_setting(key: str, value):
    with _conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO settings(key, value) VALUES(%s, %s) "
                    "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key, Json(value)))


# ---------------------------------------------------------------- admin users
def add_admin_user(uid: str, email: str, name: str, password_hash: str, created_at: str):
    with _conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO admin_users(id, email, name, password_hash, active, created_at) "
                    "VALUES(%s,%s,%s,%s,TRUE,%s)", (uid, email.lower(), name, password_hash, created_at))


def get_admin_by_email(email: str) -> dict | None:
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM admin_users WHERE lower(email)=lower(%s) AND active", (email,))
        return cur.fetchone()


def list_admin_users() -> list:
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, email, name, active, created_at, last_login "
                    "FROM admin_users ORDER BY created_at")
        rows = cur.fetchall()
    for r in rows:
        for tk in ("created_at", "last_login"):
            if r.get(tk) is not None:
                r[tk] = r[tk].isoformat()
    return rows


def count_admin_users() -> int:
    with _conn() as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM admin_users WHERE active")
        return cur.fetchone()[0]


def set_admin_last_login(uid: str, when: str):
    with _conn() as c, c.cursor() as cur:
        cur.execute("UPDATE admin_users SET last_login=%s WHERE id=%s", (when, uid))


def set_admin_password(uid: str, password_hash: str) -> bool:
    with _conn() as c, c.cursor() as cur:
        cur.execute("UPDATE admin_users SET password_hash=%s WHERE id=%s", (password_hash, uid))
        return cur.rowcount > 0


def delete_admin_user(uid: str) -> bool:
    with _conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM admin_users WHERE id=%s", (uid,))
        return cur.rowcount > 0


# ---------------------------------------------------------------- api keys
def add_api_key(kid: str, label: str, key_hash: str, prefix: str, created_at: str):
    with _conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO api_keys(id, label, key_hash, prefix, created_at, revoked_at) "
                    "VALUES(%s,%s,%s,%s,%s,NULL)", (kid, label, key_hash, prefix, created_at))


def list_api_keys() -> list:
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, label, prefix, created_at, revoked_at FROM api_keys "
                    "ORDER BY created_at DESC")
        rows = cur.fetchall()
    for r in rows:
        for tk in ("created_at", "revoked_at"):
            if r.get(tk) is not None:
                r[tk] = r[tk].isoformat()
    return rows


def active_key_hashes() -> set:
    with _conn() as c, c.cursor() as cur:
        cur.execute("SELECT key_hash FROM api_keys WHERE revoked_at IS NULL")
        return {r[0] for r in cur.fetchall()}


def revoke_api_key(kid: str, when: str) -> bool:
    with _conn() as c, c.cursor() as cur:
        cur.execute("UPDATE api_keys SET revoked_at=%s WHERE id=%s AND revoked_at IS NULL",
                    (when, kid))
        return cur.rowcount > 0


# ---------------------------------------------------------------- webhook events
def add_webhook_event(eid, task_id, event_type, target_url, payload, now):
    with _conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO webhook_events(id, task_id, event_type, target_url, status, "
                    "attempts, last_error, payload_json, created_at, updated_at) "
                    "VALUES(%s,%s,%s,%s,'pending',0,NULL,%s,%s,%s)",
                    (eid, task_id, event_type, target_url, Json(payload), now, now))


def update_webhook_event(eid, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=%s" for k in fields)
    with _conn() as c, c.cursor() as cur:
        cur.execute(f"UPDATE webhook_events SET {cols} WHERE id=%s", (*fields.values(), eid))


def deliverable_events(max_attempts: int) -> list:
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM webhook_events WHERE status='pending' "
                    "OR (status='failed' AND attempts < %s) ORDER BY created_at", (max_attempts,))
        return cur.fetchall()


def list_webhook_events(limit: int = 200, task_id: str | None = None) -> list:
    q = ("SELECT id, task_id, event_type, target_url, status, attempts, last_error, "
         "created_at, updated_at FROM webhook_events")
    args: tuple = ()
    if task_id:
        q += " WHERE task_id=%s"
        args = (task_id,)
    q += " ORDER BY created_at DESC LIMIT %s"
    args += (limit,)
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(q, args)
        rows = cur.fetchall()
    for r in rows:
        for tk in ("created_at", "updated_at"):
            if r.get(tk) is not None:
                r[tk] = r[tk].isoformat()
    return rows


def get_webhook_event(eid: str) -> dict | None:
    with _conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM webhook_events WHERE id=%s", (eid,))
        return cur.fetchone()
