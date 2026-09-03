"""Global webhook subsystem — fires task lifecycle events at the configured URL.

Distinct from the per-request `callback_url` (which still works): when the admin
enables a global webhook, every task emits an event as each stage finishes
(transcribed → summarized → analyzed). Each event is logged in `webhook_events`
and delivered by a background worker with retries + exponential backoff. The body
is HMAC-signed with the configured secret so the listener can verify authenticity.
"""
import hashlib
import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

from . import db, public, settings

MAX_ATTEMPTS = int(os.getenv("WEBHOOK_MAX_ATTEMPTS", "5"))
_RETRY_BASE = float(os.getenv("WEBHOOK_RETRY_BASE", "5"))    # seconds; doubles per attempt
_POLL = float(os.getenv("WEBHOOK_POLL", "3"))               # worker scan interval

_wake = threading.Event()
_started = False
_start_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def emit(job: dict, event_type: str):
    """Queue a lifecycle event for the global webhook, if enabled + subscribed."""
    cfg = settings.webhook()
    if not cfg.get("enabled") or not cfg.get("url"):
        return
    if event_type not in (cfg.get("events") or []):
        return
    payload = {"event": event_type, "task_id": job.get("id"), "data": public.public_result(job)}
    db.add_webhook_event(uuid.uuid4().hex[:16], job.get("id"), event_type,
                         cfg["url"], payload, _now())
    start()
    _wake.set()


def redeliver(eid: str) -> bool:
    """Manually re-arm a delivery (used by the dashboard 'refire' action)."""
    ev = db.get_webhook_event(eid)
    if not ev:
        return False
    db.update_webhook_event(eid, status="pending", last_error=None, updated_at=_now())
    start()
    _wake.set()
    return True


def _ready(ev: dict, now: float) -> bool:
    """Pending events fire immediately; failed ones wait out the backoff window."""
    if ev["status"] == "pending":
        return True
    delay = _RETRY_BASE * (2 ** max(0, ev["attempts"] - 1))
    try:
        updated = datetime.fromisoformat(ev["updated_at"]).timestamp()
    except (ValueError, TypeError):
        updated = 0
    return now >= updated + delay


def _attempt(ev: dict) -> str:
    """One delivery attempt; updates the row; returns the new status."""
    cfg = settings.webhook()
    body = json.dumps(ev["payload_json"] if isinstance(ev["payload_json"], (dict, list))
                      else json.loads(ev["payload_json"]), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "asr-tool/1.0",
               "X-ASR-Event": ev["event_type"], "X-ASR-Task": ev["task_id"] or ""}
    secret = cfg.get("secret") or ""
    if secret:
        headers["X-ASR-Signature"] = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    attempts = ev["attempts"] + 1
    try:
        req = urllib.request.Request(ev["target_url"], data=body, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            db.update_webhook_event(ev["id"], status="delivered", attempts=attempts,
                                    last_error=None, updated_at=_now())
            return "delivered"
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    # give up after MAX_ATTEMPTS; otherwise leave 'failed' for the backoff retry
    status = "failed" if attempts < MAX_ATTEMPTS else "exhausted"
    db.update_webhook_event(ev["id"], status=status, attempts=attempts,
                            last_error=err, updated_at=_now())
    return status


def _worker():
    while True:
        now = time.time()
        for ev in db.deliverable_events(MAX_ATTEMPTS):
            if _ready(ev, now):
                _attempt(ev)
        _wake.wait(timeout=_POLL)
        _wake.clear()


def start():
    """Launch the delivery worker once (idempotent)."""
    global _started
    with _start_lock:
        if _started:
            return
        threading.Thread(target=_worker, daemon=True, name="webhook-worker").start()
        _started = True
