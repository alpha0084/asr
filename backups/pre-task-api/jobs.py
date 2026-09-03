"""In-process background job runner.

A single worker thread serializes the heavy model work (one call at a time) so
we don't load Whisper/pyannote twice or blow past 16GB. Jobs live in memory.

Two-step flow: the transcribe pass runs first; summary and QA analytics are
triggered on demand per job (web UI), or inline (public API / CLI).
"""
import json
import threading
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import public
from .pipeline import runner

_executor = ThreadPoolExecutor(max_workers=1)   # serialize heavy work
_jobs: dict = {}
_lock = threading.Lock()


def create_job(src, filename, model=None, language=None, speakers=None, target_language=None,
               include_summary=False, include_analytics=False, callback_url=None) -> str:
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = {
            "id": jid, "filename": filename, "status": "queued",
            "stage": None, "detail": None, "stages": runner.STAGES,
            "result": None, "error": None,
            "summary_status": None, "analytics_status": None,
        }
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

    def on_progress(detail):
        with _lock:
            _jobs[jid]["detail"] = detail

    try:
        result = runner.transcribe_only(
            src, model=model, language=language, speakers=speakers,
            target_language=target_language, on_stage=on_stage, on_progress=on_progress)
        with _lock:
            _jobs[jid]["result"] = result
    except Exception as e:
        with _lock:
            _jobs[jid].update(status="error", error=f"{type(e).__name__}: {e}")
        if callback_url:
            _deliver(jid, callback_url)
        return

    # Inline stages for the all-in-one API path — keep status "running" until done.
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
    if callback_url:
        _deliver(jid, callback_url)


# ---- on-demand stages (web UI: Summarize / Analyze buttons) ----
def trigger_stage(jid, name) -> bool:
    with _lock:
        job = _jobs.get(jid)
        if not job or job.get("status") != "done" or not job.get("result"):
            return False
        key = "summary_status" if name == "summary" else "analytics_status"
        if job.get(key) == "running":
            return True                      # already in progress
        job[key] = "queued"
    _executor.submit(_do_summary if name == "summary" else _do_analytics, jid)
    return True


def _do_summary(jid):
    with _lock:
        _jobs[jid]["summary_status"] = "running"
    def on_stage(_): pass
    try:
        with _lock:
            result = _jobs[jid]["result"]
        runner.add_summary(result)
        with _lock:
            _jobs[jid]["summary_status"] = "done"
    except Exception as e:
        with _lock:
            _jobs[jid]["summary_status"] = f"error: {e}"


def _do_analytics(jid):
    with _lock:
        _jobs[jid]["analytics_status"] = "running"
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
    except Exception as e:
        with _lock:
            _jobs[jid]["analytics_status"] = f"error: {e}"


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
