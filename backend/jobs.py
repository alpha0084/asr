"""Durable, concurrent job runner backed by Postgres.

Recordings are enqueued to Postgres (status='queued'); a pool of worker threads
claims them with `SELECT … FOR UPDATE SKIP LOCKED` and runs the pipeline. Because
the queue lives in the DB, queued work survives restarts, and extra worker
machines can pull from the same `DATABASE_URL` — scale out by adding workers/GPUs
with no code change. Live progress (stage/detail) is kept in memory and merged
into status reads; the durable state (queued/running/done + result) is in Postgres.
"""
import json
import os
import shutil
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import db, public, webhooks
from .config import DATA_DIR, WORKER_CONCURRENCY
from .pipeline import runner

_live: dict = {}                 # {jid: {"stage":..., "detail":...}} transient progress
_live_lock = threading.Lock()
_wake = threading.Event()        # nudges idle workers when new work is enqueued
_started = False
_start_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set_live(jid, **kw):
    with _live_lock:
        _live.setdefault(jid, {}).update(kw)


def _clear_live(jid):
    with _live_lock:
        _live.pop(jid, None)


def _record_run(jid, run_type, params, result):
    try:
        db.record_run(jid, run_type, params, result)
    except Exception as e:
        print(f"[db] record_run({run_type}) failed for {jid}: {e}", flush=True)


# ---------------------------------------------------------------- public API
def create_job(src, filename, model=None, language=None, speakers=None, target_language=None,
               include_summary=False, include_analytics=False, callback_url=None) -> str:
    """Enqueue a recording. Returns immediately with a job id; workers process it."""
    jid = uuid.uuid4().hex[:12]
    final_stage = ("analytics" if include_analytics else
                   "summary" if include_summary else "transcribe")
    params = {
        "model": model, "language": language, "speakers": speakers,
        "target_language": target_language, "callback_url": callback_url,
        "include_summary": include_summary, "include_analytics": include_analytics,
        "final_stage": final_stage,
    }
    db.enqueue_recording(jid, filename, src, params)
    start()
    _wake.set()
    return jid


def trigger_stage(jid, name) -> bool:
    """Queue an on-demand stage (summary|analytics) on a finished recording."""
    job = db.load_one(jid)
    if not job or job.get("status") != "done" or not job.get("result"):
        return False
    col = "summary_status" if name == "summary" else "analytics_status"
    if job.get(col) == "running":
        return True
    db.set_recording(jid, **{col: "queued"})
    start()
    _wake.set()
    return True


def get_job(jid) -> dict:
    job = db.load_one(jid) or {}
    if job:
        with _live_lock:
            live = _live.get(jid)
        if live:
            job = {**job, **{k: v for k, v in live.items() if v is not None}}
        job.setdefault("stages", runner.STAGES)
    return job


def list_jobs() -> list:
    rows = db.list_recordings()
    with _live_lock:
        live = dict(_live)
    for r in rows:
        if r["id"] in live:
            r.update({k: v for k, v in live[r["id"]].items() if v is not None})
    return rows


def queue_status() -> dict:
    c = db.queue_counts()
    with _live_lock:
        running = [{"id": jid, "stage": v.get("stage"), "detail": v.get("detail")}
                   for jid, v in _live.items()]
    return {"workers": WORKER_CONCURRENCY,
            "counts": {"queued": c["queued"], "running": c["running"]},
            "queued": [], "running": running, "active_stages": c["active_stages"]}


def retranscribe(jid: str) -> bool:
    job = db.load_one(jid)
    if not job or not job.get("source"):
        return False
    db.set_recording(jid, status="queued", stage=None, error=None, result=None,
                     summary_status=None, analytics_status=None)
    start()
    _wake.set()
    return True


def delete_job(jid: str) -> bool:
    job = db.load_one(jid)
    result = (job or {}).get("result") or {}
    out_json = result.get("out_json")
    if out_json and _within(out_json):
        shutil.rmtree(Path(out_json).parent, ignore_errors=True)
    src = result.get("input")
    if src and _within(src) and os.path.isfile(src):
        try:
            os.remove(src)
        except OSError:
            pass
    db.delete_task(jid)
    _clear_live(jid)
    return job is not None


def _within(path: str) -> bool:
    try:
        Path(path).resolve().relative_to(DATA_DIR.resolve())
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- worker pool
def load_from_db():
    """Startup: ensure schema, put interrupted work back on the queue, start workers."""
    db.init_db()
    db.requeue_running()
    start()


def start():
    """Idempotently launch the worker pool."""
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    for i in range(max(1, WORKER_CONCURRENCY)):
        threading.Thread(target=_worker_loop, name=f"asr-worker-{i}", daemon=True).start()
    print(f"[jobs] worker pool started: {WORKER_CONCURRENCY} workers", flush=True)


def _worker_loop():
    while True:
        try:
            did = _process_one()
        except Exception as e:
            print(f"[worker] error: {e}", flush=True)
            did = False
        if not did:
            _wake.wait(timeout=2.0)
            _wake.clear()


def _process_one() -> bool:
    claim = db.claim_transcribe()
    if claim:
        _run_transcribe(claim)
        return True
    claim = db.claim_stage("summary")
    if claim:
        _run_stage(claim, "summary")
        return True
    claim = db.claim_stage("analytics")
    if claim:
        _run_stage(claim, "analytics")
        return True
    return False


def _run_transcribe(claim):
    jid = claim["id"]
    src = claim["source"]
    p = claim.get("request_params") or {}

    def on_stage(name):
        _set_live(jid, stage=name, detail=None)
        db.set_recording(jid, stage=name)

    def on_progress(detail):
        _set_live(jid, detail=detail)

    try:
        result = runner.transcribe_only(
            src, model=p.get("model"), language=p.get("language"),
            speakers=p.get("speakers"), target_language=p.get("target_language"),
            on_stage=on_stage, on_progress=on_progress, work_id=jid)
    except Exception as e:
        db.set_recording(jid, status="error", error=f"{type(e).__name__}: {e}")
        _clear_live(jid)
        _deliver(jid, p.get("callback_url"))
        return

    db.set_recording(jid, status="done", stage="Done", result=result)
    _clear_live(jid)
    _record_run(jid, "transcribe", result.get("params"), result)
    webhooks.emit(get_job(jid), "transcribed")

    # All-in-one path: chain the requested on-demand stages onto the queue.
    if p.get("include_summary"):
        db.set_recording(jid, summary_status="queued")
        _wake.set()
    elif p.get("include_analytics"):
        db.set_recording(jid, analytics_status="queued")
        _wake.set()
    if p.get("final_stage") == "transcribe":
        _deliver(jid, p.get("callback_url"))


def _run_stage(claim, stage):
    jid = claim["id"]
    result = claim.get("result_json") or {}
    p = claim.get("request_params") or {}
    _set_live(jid, stage=("Summarizing" if stage == "summary" else "Analyzing"))
    try:
        if stage == "summary":
            runner.add_summary(result)
            params, event = result.get("summary_params"), "summarized"
        else:
            runner.add_analytics(result, on_progress=lambda d: _set_live(jid, detail=d))
            params, event = result.get("analytics_params"), "analyzed"
        db.set_recording(jid, result=result, **{f"{stage}_status": "done"})
        _clear_live(jid)
        _record_run(jid, stage, params, result)
        webhooks.emit(get_job(jid), event)
    except Exception as e:
        db.set_recording(jid, **{f"{stage}_status": f"error: {e}"})
        _clear_live(jid)
        return

    # Chain analytics after summary for the all-in-one path; fire callback on the last stage.
    if stage == "summary" and p.get("include_analytics"):
        db.set_recording(jid, analytics_status="queued")
        _wake.set()
    if p.get("final_stage") == stage:
        _deliver(jid, p.get("callback_url"))


# ---------------------------------------------------------------- per-request callback
def _deliver(jid, callback_url):
    if not callback_url:
        return
    _post_callback(callback_url, public.public_result(get_job(jid)))


def _post_callback(url, payload, retries=2) -> str:
    data = json.dumps(payload).encode("utf-8")
    last = "not attempted"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json", "User-Agent": "asr-tool/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"[callback] POST {url} -> {resp.status}", flush=True)
                return f"delivered ({resp.status})"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} from your server"
            print(f"[callback] attempt {attempt+1}: {last}", flush=True)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            print(f"[callback] attempt {attempt+1} failed: {last}", flush=True)
    return f"failed — {last}"
