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


def _target_for_task(tid: str) -> dict | None:
    """Resolve where THIS recording's webhooks go, and with what secret/events.

    Priority: per-request `callback_url` > the submitting API key's webhook > the
    global webhook. This is what lets one shared ASR tool fan results out to
    multiple environments (a key/webhook per environment). Returns None if nothing
    is configured.
    """
    params = db.get_request_params(tid) or {}
    cb = (params.get("callback_url") or "").strip()
    if cb:
        return {"url": cb, "secret": params.get("callback_secret") or "",
                "events": ["transcribed", "summarized", "analyzed", "error"]}
    kid = params.get("api_key_id")
    if kid:
        kw = db.get_key_webhook(kid)
        if kw:
            return kw
    g = settings.webhook()
    if g.get("enabled") and g.get("url"):
        return {"url": g["url"], "secret": g.get("secret") or "", "events": g.get("events") or []}
    return None


def emit(job: dict, event_type: str):
    """Queue a lifecycle event for delivery to the recording's resolved target."""
    tid = job.get("id")
    target = _target_for_task(tid)
    if not target or not target.get("url"):
        return
    if event_type not in (target.get("events") or []):
        return
    payload = {"event": event_type, "task_id": tid, "data": public.public_result(job)}
    db.add_webhook_event(uuid.uuid4().hex[:16], tid, event_type, target["url"], payload, _now())
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
    # Re-resolve the target so we sign with the RIGHT secret for this recording's
    # environment (per-request / per-key / global). URL is fixed at emit time.
    target = _target_for_task(ev["task_id"]) or {}
    body = json.dumps(ev["payload_json"] if isinstance(ev["payload_json"], (dict, list))
                      else json.loads(ev["payload_json"]), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "asr-tool/1.0",
               "X-ASR-Event": ev["event_type"], "X-ASR-Task": ev["task_id"] or ""}
    secret = target.get("secret") or ""
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
