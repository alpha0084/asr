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

# Live-resizable pool. `_desired` is the LIVE target worker count (may be autoscaled above the
# baseline); `_live_count` is how many worker threads are actually alive. Growing spawns threads
# immediately; shrinking lets surplus workers retire after their current job. `_baseline` is the
# admin-set resting floor autoscaling returns to; `_autoscale` toggles the autoscaler.
_desired: int | None = None
_live_count = 0
_pool_lock = threading.Lock()
_baseline: int | None = None
_autoscale = False
_autoscale_stop = threading.Event()


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
               include_summary=False, include_analytics=False, callback_url=None,
               api_key_id=None) -> str:
    """Enqueue a recording. Returns immediately with a job id; workers process it.

    `api_key_id` tags the recording with the key that submitted it, so lifecycle
    webhooks fan out to THAT key's configured listener — this is how one shared ASR
    tool serves multiple environments (staging/prod) with a key per environment.
    `callback_url` (optional) overrides that with a per-request target.
    """
    jid = uuid.uuid4().hex[:12]
    final_stage = ("analytics" if include_analytics else
                   "summary" if include_summary else "transcribe")
    params = {
        "model": model, "language": language, "speakers": speakers,
        "target_language": target_language, "callback_url": callback_url,
        "api_key_id": api_key_id,
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


def list_jobs(limit=None, offset=0, date_from=None, date_to=None, status=None, search=None) -> list:
    rows = db.list_recordings(limit=limit, offset=offset, date_from=date_from,
                              date_to=date_to, status=status, search=search)
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


def current_concurrency() -> int:
    """The LIVE target worker count (may be autoscaled above the baseline). Falls back to the boot
    default before start() runs."""
    return _desired if _desired is not None else max(1, WORKER_CONCURRENCY)


def current_baseline() -> int:
    """The admin-set resting concurrency — the floor autoscaling returns to when idle."""
    return _baseline if _baseline is not None else current_concurrency()


def autoscale_enabled() -> bool:
    return bool(_autoscale)


def _spawn(n: int) -> None:
    """Start n worker threads. Caller must hold _pool_lock."""
    global _live_count
    for _ in range(max(0, n)):
        _live_count += 1
        threading.Thread(target=_worker_loop, name=f"asr-worker-{_live_count}",
                         daemon=True).start()


def _resize(n: int) -> int:
    """Set the LIVE target worker count — grow instantly, shrink by retiring surplus workers between
    jobs. Does NOT persist; used by both the manual setter and the autoscaler."""
    global _desired
    n = max(1, int(n))
    with _pool_lock:
        _desired = n
        if _live_count < n:
            _spawn(n - _live_count)
        # if _live_count > n, the extra workers exit themselves at the top of _worker_loop
    _wake.set()
    return n


def set_concurrency(n: int) -> int:
    """Admin action: set the resting BASELINE concurrency, persist it, and resize live to it.
    Autoscaling (when on) may still burst above this under load. Headroom validation is the
    caller's job (see admin_api / metrics.headroom)."""
    global _baseline
    n = max(1, int(n))
    _baseline = n
    try:
        db.set_setting("worker_concurrency", n)
    except Exception as e:
        print(f"[jobs] persist worker_concurrency failed: {e}", flush=True)
    _resize(n)
    print(f"[jobs] worker concurrency baseline set to {n}", flush=True)
    return n


def set_autoscale(on: bool) -> bool:
    """Enable/disable autoscaling and persist. Turning it off settles the pool back to the baseline."""
    global _autoscale
    _autoscale = bool(on)
    try:
        db.set_setting("autoscale_enabled", _autoscale)
    except Exception as e:
        print(f"[jobs] persist autoscale_enabled failed: {e}", flush=True)
    if not _autoscale and _baseline is not None:
        _resize(_baseline)       # drop any burst back to the resting floor
    print(f"[jobs] autoscale {'enabled' if _autoscale else 'disabled'}", flush=True)
    return _autoscale


def _autoscale_ceiling() -> int:
    """Highest live worker count autoscaling may use right now: bounded by AUTOSCALE_MAX and the
    real-time VRAM/RAM headroom, so a burst never overcommits the shared GPU."""
    from . import metrics
    from .config import AUTOSCALE_MAX
    hr = metrics.headroom()
    return max(current_baseline(), min(AUTOSCALE_MAX, hr.get("max_safe", current_concurrency())))


def _autoscaler_loop():
    """Burst the pool toward the backlog size when work piles up; settle back to the baseline after
    an idle cooldown. A no-op while autoscaling is off."""
    import time
    from .config import AUTOSCALE_INTERVAL_SEC, AUTOSCALE_COOLDOWN_SEC, AUTOSCALE_STEP
    idle_since = None
    while not _autoscale_stop.wait(AUTOSCALE_INTERVAL_SEC):
        try:
            if not _autoscale:
                idle_since = None
                continue
            counts = db.queue_counts()
            demand = counts.get("queued", 0) + counts.get("running", 0)
            cur = current_concurrency()
            if demand > cur:
                # scale UP toward the backlog, capped by the live headroom ceiling — drain fast
                target = min(demand, _autoscale_ceiling())
                if target > cur:
                    _resize(target)
                    print(f"[autoscale] backlog {demand} → {target} workers", flush=True)
                idle_since = None
            elif demand == 0 and cur > current_baseline():
                # idle → settle back to the baseline after a cooldown (avoids flapping)
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since >= AUTOSCALE_COOLDOWN_SEC:
                    target = max(current_baseline(), cur - AUTOSCALE_STEP)
                    _resize(target)
                    print(f"[autoscale] idle → {target} workers", flush=True)
                    idle_since = None if target <= current_baseline() else time.time()
            else:
                idle_since = None
        except Exception as e:
            print(f"[autoscale] error: {e}", flush=True)


def start():
    """Idempotently launch the worker pool + autoscaler, sized from persisted settings."""
    global _started, _baseline, _autoscale
    with _start_lock:
        if _started:
            return
        _started = True
    from .config import AUTOSCALE_ENABLED_DEFAULT
    persisted = None
    try:
        persisted = db.get_setting("worker_concurrency")
    except Exception:
        pass
    _baseline = max(1, int(persisted)) if persisted else max(1, WORKER_CONCURRENCY)
    try:
        a = db.get_setting("autoscale_enabled")
        _autoscale = AUTOSCALE_ENABLED_DEFAULT if a is None else bool(a)
    except Exception:
        _autoscale = AUTOSCALE_ENABLED_DEFAULT
    _resize(_baseline)
    threading.Thread(target=_autoscaler_loop, name="asr-autoscaler", daemon=True).start()
    print(f"[jobs] worker pool started: {_baseline} workers "
          f"(autoscale={'on' if _autoscale else 'off'})", flush=True)


def _worker_loop():
    global _live_count
    while True:
        # Honour a concurrency decrease: a surplus worker retires here, between jobs, so an
        # in-flight transcription is never interrupted.
        with _pool_lock:
            if _desired is not None and _live_count > _desired:
                _live_count -= 1
                print(f"[jobs] worker retired → {_live_count} live", flush=True)
                return
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
        webhooks.emit(get_job(jid), "error")
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

    # Chain analytics after summary for the all-in-one path.
    if stage == "summary" and p.get("include_analytics"):
        db.set_recording(jid, analytics_status="queued")
        _wake.set()
