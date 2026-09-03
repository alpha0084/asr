"""In-process background job runner.

A single worker thread serializes the heavy model work (one call at a time) so
we don't load Whisper/pyannote twice or blow past 16GB. Jobs live in memory.
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


def create_job(src, filename, model=None, language=None, speakers=None,
               target_language=None, analytics=True, callback_url=None) -> str:
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = {
            "id": jid, "filename": filename, "status": "queued",
            "stage": None, "detail": None, "stages": runner.STAGES,
            "result": None, "error": None,
        }
    _executor.submit(_run, jid, src, model, language, speakers,
                     target_language, analytics, callback_url)
    return jid


def _run(jid, src, model, language, speakers, target_language, analytics, callback_url=None):
    def on_stage(name):
        with _lock:
            _jobs[jid]["status"] = "running"
            _jobs[jid]["stage"] = name
            _jobs[jid]["detail"] = None

    def on_progress(detail):
        with _lock:
            _jobs[jid]["detail"] = detail

    try:
        result = runner.process(src, model=model, language=language, speakers=speakers,
                                target_language=target_language, analytics=analytics,
                                on_stage=on_stage, on_progress=on_progress)
        with _lock:
            _jobs[jid].update(status="done", stage="Done", result=result)
    except Exception as e:
        with _lock:
            _jobs[jid].update(status="error", error=f"{type(e).__name__}: {e}")

    if callback_url:
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
