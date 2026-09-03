"""In-process background job runner with SQLite persistence.

A single worker thread serializes the heavy model work (one call at a time). Job
state is mirrored to SQLite so task history + results survive server restarts.
Transcribe runs first; summary and QA analytics are triggered on demand per task.
"""
import json
import os
import shutil
import threading
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import db, public, webhooks
from .config import DATA_DIR
from .pipeline import runner

_executor = ThreadPoolExecutor(max_workers=1)   # serialize heavy work
_jobs: dict = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _persist(jid):
    """Mirror the in-memory job to SQLite (called on state transitions)."""
    with _lock:
        job = dict(_jobs.get(jid) or {})
    if job:
        job["updated_at"] = _now()
        db.save(job)


def _record_run(jid, run_type, params, result):
    """Store an immutable versioned run (never blocks the job on a DB error)."""
    try:
        db.record_run(jid, run_type, params, result)
    except Exception as e:
        print(f"[db] record_run({run_type}) failed for {jid}: {e}", flush=True)


def load_from_db():
    """On startup: repopulate tasks from the DB; mark unfinished ones interrupted."""
    db.init_db()
    for job in db.load_all():
        if job.get("status") in ("queued", "running"):
            job["status"] = "interrupted"
            job["error"] = job.get("error") or "server restarted while processing"
        job.setdefault("detail", None)
        _jobs[job["id"]] = job
    # persist the interrupted transitions
    for jid, job in list(_jobs.items()):
        if job.get("status") == "interrupted":
            _persist(jid)


def create_job(src, filename, model=None, language=None, speakers=None, target_language=None,
               include_summary=False, include_analytics=False, callback_url=None) -> str:
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = {
            "id": jid, "filename": filename, "source": src, "status": "queued",
            "stage": None, "detail": None, "stages": runner.STAGES,
            "result": None, "error": None,
            "summary_status": None, "analytics_status": None,
            "created_at": _now(),
        }
    _persist(jid)
    _executor.submit(_run, jid, src, model, language, speakers, target_language,
                     include_summary, include_analytics, callback_url)
    return jid


def _run(jid, src, model, language, speakers, target_language,
         include_summary, include_analytics, callback_url=None):
    def on_stage(name):
        with _lock:
            _jobs[jid]["status"] = "running"
            _jobs[jid]["stage"] = name
            _jobs[jid]["detail"] = None
        _persist(jid)

    def on_progress(detail):
        with _lock:
            _jobs[jid]["detail"] = detail

    try:
        result = runner.transcribe_only(
            src, model=model, language=language, speakers=speakers,
            target_language=target_language, on_stage=on_stage, on_progress=on_progress)
        with _lock:
            _jobs[jid]["result"] = result
        _persist(jid)
        _record_run(jid, "transcribe", result.get("params"), result)
        webhooks.emit(get_job(jid), "transcribed")
    except Exception as e:
        with _lock:
            _jobs[jid].update(status="error", error=f"{type(e).__name__}: {e}")
        _persist(jid)
        if callback_url:
            _deliver(jid, callback_url)
        return

    if include_summary:
        with _lock:
            _jobs[jid]["stage"] = "Summarizing"
        _do_summary(jid)
    if include_analytics:
        with _lock:
            _jobs[jid]["stage"] = "Analyzing"
        _do_analytics(jid)

    with _lock:
        _jobs[jid].update(status="done", stage="Done")
    _persist(jid)
    if callback_url:
        _deliver(jid, callback_url)


# ---- on-demand stages (Summarize / Analyze) ----
def trigger_stage(jid, name) -> bool:
    with _lock:
        job = _jobs.get(jid)
        if not job or job.get("status") != "done" or not job.get("result"):
            return False
        key = "summary_status" if name == "summary" else "analytics_status"
        if job.get(key) == "running":
            return True
        job[key] = "queued"
    _persist(jid)
    _executor.submit(_do_summary if name == "summary" else _do_analytics, jid)
    return True


def _do_summary(jid):
    with _lock:
        _jobs[jid]["summary_status"] = "running"
    _persist(jid)
    try:
        with _lock:
            result = _jobs[jid]["result"]
        runner.add_summary(result)
        with _lock:
            _jobs[jid]["summary_status"] = "done"
            _jobs[jid]["result"] = result
        _persist(jid)
        _record_run(jid, "summary", result.get("summary_params"), result)
        webhooks.emit(get_job(jid), "summarized")
        return
    except Exception as e:
        with _lock:
            _jobs[jid]["summary_status"] = f"error: {e}"
    _persist(jid)


def _do_analytics(jid):
    with _lock:
        _jobs[jid]["analytics_status"] = "running"
    _persist(jid)
    def on_progress(detail):
        with _lock:
            _jobs[jid]["detail"] = detail
    try:
        with _lock:
            result = _jobs[jid]["result"]
        runner.add_analytics(result, on_progress=on_progress)
        with _lock:
            _jobs[jid]["analytics_status"] = "done"
            _jobs[jid]["detail"] = None
            _jobs[jid]["result"] = result
        _persist(jid)
        _record_run(jid, "analytics", result.get("analytics_params"), result)
        webhooks.emit(get_job(jid), "analyzed")
        return
    except Exception as e:
        with _lock:
            _jobs[jid]["analytics_status"] = f"error: {e}"
    _persist(jid)


def _deliver(jid, callback_url):
    status = _post_callback(callback_url, public.public_result(get_job(jid)))
    with _lock:
        _jobs[jid]["callback_result"] = status


def _post_callback(url, payload, retries=2) -> str:
    data = json.dumps(payload).encode("utf-8")
    last = "not attempted"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json", "User-Agent": "asr-tool/1.0"},
            )
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


def get_job(jid) -> dict:
    with _lock:
        return dict(_jobs.get(jid) or {})


def list_jobs() -> list:
    with _lock:
        return [{k: v for k, v in j.items() if k != "result"} for j in _jobs.values()]


# ---- dashboard: queue monitor + task actions ----
def queue_status() -> dict:
    """Snapshot of the single heavy-work queue + any on-demand stages in flight."""
    with _lock:
        snap = list(_jobs.values())
    queued = [j["id"] for j in snap if j.get("status") == "queued"]
    running = [{"id": j["id"], "stage": j.get("stage"), "detail": j.get("detail")}
               for j in snap if j.get("status") == "running"]
    active_stages = [
        {"id": j["id"], "summary_status": j.get("summary_status"),
         "analytics_status": j.get("analytics_status")}
        for j in snap
        if j.get("summary_status") in ("queued", "running")
        or j.get("analytics_status") in ("queued", "running")]
    return {"workers": 1, "counts": {"queued": len(queued), "running": len(running)},
            "queued": queued, "running": running, "active_stages": active_stages}


def retranscribe(jid: str) -> bool:
    """Re-run the transcribe pass in place on the stored source (defaults for options)."""
    job = get_job(jid)
    if not job or not job.get("source"):
        return False
    with _lock:
        _jobs[jid].update(status="queued", stage=None, detail=None, error=None,
                          result=None, summary_status=None, analytics_status=None)
    _persist(jid)
    _executor.submit(_run, jid, job["source"], None, None, None, None, False, False, None)
    return True


def _within(path: str) -> bool:
    """True if `path` lives under DATA_DIR — a guard before deleting anything."""
    try:
        Path(path).resolve().relative_to(DATA_DIR.resolve())
        return True
    except (ValueError, TypeError):
        return False


def delete_job(jid: str) -> bool:
    with _lock:
        job = _jobs.pop(jid, None)
    result = (job or {}).get("result") or {}
    # remove the per-call work dir (audio16k.wav + transcript.json) and any local source
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
    return job is not None
