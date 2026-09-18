"""Admin portal API — auth + runtime configuration.

Mounted under /admin/api. Everything except login/status requires the admin
session (see admin_auth.require_admin). Dashboard, webhook-log, and task actions
are added in later phases.
"""
import secrets
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from pydantic import BaseModel

from . import admin_auth, db, jobs, metrics, public, settings, webhooks
from .settings import K_MODELS, K_SCORECARD, K_SCORECARD_VERSION, K_SUMMARY, K_WEBHOOK


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

router = APIRouter(prefix="/admin/api", tags=["Admin"])

_ADMIN = [Depends(admin_auth.require_admin)]


class LoginBody(BaseModel):
    email: str
    password: str


class AdminBody(BaseModel):
    email: str
    password: str
    name: str = ""


class ModelsBody(BaseModel):
    transcribe_model: str = ""
    asr_backend: str = ""
    speaker_mode: str = ""
    diarize_model: str = ""
    summary_model: str = ""
    analytics_model: str = ""
    target_language: str = ""


# ---------------------------------------------------------------- auth
@router.get("/status", summary="Is admin configured / am I logged in?")
def status():
    return {"configured": admin_auth.is_configured()}


@router.post("/login", summary="Admin login (email + password) → session token + cookie")
def login(body: LoginBody, response: Response):
    res = admin_auth.login(body.email, body.password)
    if not res:
        raise HTTPException(401, "invalid email or password")
    # httponly cookie for the browser UI; token also returned for API clients.
    response.set_cookie("admin_session", res["token"], httponly=True, samesite="lax", max_age=12 * 3600)
    return {"token": res["token"], "user": {"email": res["email"], "name": res["name"]}}


@router.post("/logout", dependencies=_ADMIN, summary="Log out (invalidate session)")
def logout(response: Response, me: dict = Depends(admin_auth.require_admin)):
    admin_auth.logout(me["token"])
    response.delete_cookie("admin_session")
    return {"ok": True}


@router.get("/me", summary="Current logged-in admin")
def me(me: dict = Depends(admin_auth.require_admin)):
    return {"email": me["email"], "name": me["name"]}


# ---------------------------------------------------------------- admin users
@router.get("/admins", dependencies=_ADMIN, summary="List admin users")
def list_admins():
    return db.list_admin_users()


@router.post("/admins", dependencies=_ADMIN, summary="Create an admin user")
def create_admin(body: AdminBody):
    try:
        return admin_auth.create_admin(body.email, body.password, body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/admins/{uid}", summary="Delete an admin (cannot delete the last one or yourself)")
def delete_admin(uid: str, me: dict = Depends(admin_auth.require_admin)):
    if uid == me["id"]:
        raise HTTPException(400, "you cannot delete your own account while logged in")
    if db.count_admin_users() <= 1:
        raise HTTPException(400, "cannot delete the last admin")
    if not db.delete_admin_user(uid):
        raise HTTPException(404, "admin not found")
    return {"ok": True}


# ---------------------------------------------------------------- live metrics
@router.get("/metrics", dependencies=_ADMIN, summary="Live system metrics (GPU/RAM/models/traffic/workers)")
def live_metrics():
    return metrics.snapshot()


# ---------------------------------------------------------------- worker concurrency
class ConcurrencyBody(BaseModel):
    concurrency: int


@router.get("/settings/concurrency", dependencies=_ADMIN,
            summary="Current worker concurrency + how high it can safely go right now")
def get_concurrency():
    return metrics.headroom()


@router.put("/settings/concurrency", dependencies=_ADMIN,
            summary="Set worker concurrency (validated against live VRAM/RAM headroom)")
def put_concurrency(body: ConcurrencyBody):
    n = int(body.concurrency)
    if n < 1:
        raise HTTPException(status_code=400, detail={"ok": False, "message": "Concurrency must be at least 1."})
    hr = metrics.headroom()
    if n > hr["max_safe"]:
        # Not enough headroom — reject and tell the admin the safe ceiling. Nothing is changed.
        raise HTTPException(status_code=409, detail={
            "ok": False, "requested": n, "max_safe": hr["max_safe"], "limited_by": hr["limited_by"],
            "message": (f"Not enough {hr['limited_by']} headroom for {n} workers. "
                        f"The most you can safely run right now is {hr['max_safe']}. "
                        f"Use {hr['max_safe']} or fewer."),
            "headroom": hr,
        })
    applied = jobs.set_concurrency(n)
    return {"ok": True, "concurrency": applied, "headroom": metrics.headroom()}


class AutoscaleBody(BaseModel):
    enabled: bool


@router.put("/settings/autoscale", dependencies=_ADMIN,
            summary="Enable/disable autoscaling (burst workers up on a backlog, settle back when idle)")
def put_autoscale(body: AutoscaleBody):
    on = jobs.set_autoscale(bool(body.enabled))
    return {"ok": True, "autoscale": on, "headroom": metrics.headroom()}


# ---------------------------------------------------------------- config: models/backend
_WHISPER_OPTS = [
    {"id": "large-v3-turbo", "speed": "fastest · great accuracy (recommended)"},
    {"id": "distil-large-v3.5", "speed": "very fast · good accuracy"},
    {"id": "medium", "speed": "fast · decent accuracy"},
    {"id": "small", "speed": "faster · lower accuracy"},
    {"id": "base", "speed": "very fast · low accuracy"},
    {"id": "tiny", "speed": "fastest · lowest accuracy"},
    {"id": "large-v3", "speed": "slowest · best accuracy"},
]
_SPEAKER_MODES = [
    {"id": "llm", "desc": "LLM per-line — best for mono/phone, finest splits"},
    {"id": "hybrid", "desc": "acoustic clusters + LLM role naming"},
    {"id": "acoustic", "desc": "pyannote only — generic Speaker 1/2, no roles"},
]


def _ollama_models() -> list:
    import json
    import os
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=5) as r:
            data = json.load(r)
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


@router.get("/settings/models", dependencies=_ADMIN, summary="Per-process model config + options")
def get_models():
    return {
        "current": settings.models(),
        "available": {
            "whisper": _WHISPER_OPTS,
            "backends": ["cuda", "faster", "mlx"],
            "speaker_modes": _SPEAKER_MODES,
            "ollama": _ollama_models(),
        },
    }


@router.put("/settings/models", dependencies=_ADMIN, summary="Update model/backend config")
def put_models(body: ModelsBody):
    # store only the keys the admin set; blanks fall back to .env defaults via settings.models()
    db.set_setting(K_MODELS, {k: v.strip() for k, v in body.model_dump().items() if v.strip()})
    return settings.models()


# ---------------------------------------------------------------- config: QA scorecard
@router.get("/settings/scorecard", dependencies=_ADMIN, summary="QA scorecard (categories/checkpoints/weights)")
def get_scorecard():
    return {"scorecard": settings.scorecard(), "default": settings.DEFAULT_SCORECARD}


@router.put("/settings/scorecard", dependencies=_ADMIN, summary="Replace the QA scorecard")
def put_scorecard(scorecard: list = Body(..., embed=True)):
    _validate_scorecard(scorecard)
    # Preserve ids the editor sent (renames keep their id); mint ids for new items.
    scorecard, _ = settings._ensure_ids(scorecard)
    db.set_setting(K_SCORECARD, scorecard)
    db.set_setting(K_SCORECARD_VERSION, settings.scorecard_version() + 1)
    return {"scorecard": settings.scorecard(), "scorecard_version": settings.scorecard_version()}


def _validate_scorecard(sc):
    if not isinstance(sc, list) or not sc:
        raise HTTPException(400, "scorecard must be a non-empty list of categories")
    for cat in sc:
        if not isinstance(cat, dict) or not str(cat.get("category", "")).strip():
            raise HTTPException(400, "each category needs a non-empty 'category' name")
        cps = cat.get("checkpoints")
        if not isinstance(cps, list) or not cps:
            raise HTTPException(400, f"category '{cat.get('category')}' needs checkpoints")
        for cp in cps:
            if not isinstance(cp, dict) or not str(cp.get("name", "")).strip():
                raise HTTPException(400, "each checkpoint needs a non-empty 'name'")


# ---------------------------------------------------------------- config: summary fields
@router.get("/settings/summary", dependencies=_ADMIN, summary="Summary-stage field config")
def get_summary():
    return {"summary": settings.summary_config(), "default": settings.DEFAULT_SUMMARY}


@router.put("/settings/summary", dependencies=_ADMIN, summary="Update summary-stage fields")
def put_summary(summary: dict = Body(..., embed=True)):
    db.set_setting(K_SUMMARY, summary)
    return {"summary": settings.summary_config()}


# ---------------------------------------------------------------- config: webhook
@router.get("/settings/webhook", dependencies=_ADMIN, summary="Global webhook config")
def get_webhook():
    return settings.webhook()


@router.put("/settings/webhook", dependencies=_ADMIN, summary="Update global webhook config")
def put_webhook(webhook: dict = Body(..., embed=True)):
    cur = settings.webhook()
    db.set_setting(K_WEBHOOK, {**cur, **webhook})
    return settings.webhook()


# ---------------------------------------------------------------- X-API-Keys
class KeyBody(BaseModel):
    label: str = ""


@router.get("/keys", dependencies=_ADMIN, summary="List issued API keys (no secrets)")
def list_keys():
    return db.list_api_keys()


@router.post("/keys", dependencies=_ADMIN, summary="Create an API key (secret shown once)")
def create_key(body: KeyBody):
    raw = "asr_" + secrets.token_urlsafe(24)
    kid = uuid.uuid4().hex[:12]
    db.add_api_key(kid, body.label.strip() or "unnamed",
                   admin_auth.hash_api_key(raw), raw[:12], _now())
    # The plaintext key is returned exactly once — it is never stored.
    return {"id": kid, "label": body.label.strip() or "unnamed", "key": raw, "prefix": raw[:12]}


@router.delete("/keys/{kid}", dependencies=_ADMIN, summary="Revoke an API key")
def revoke_key(kid: str):
    if not db.revoke_api_key(kid, _now()):
        raise HTTPException(404, "key not found or already revoked")
    return {"ok": True}


class KeyWebhookBody(BaseModel):
    url: str = ""
    secret: str = ""
    events: list = []


@router.put("/keys/{kid}/webhook", dependencies=_ADMIN,
            summary="Set a key's webhook target (routes that environment's results)")
def set_key_webhook(kid: str, body: KeyWebhookBody):
    events = [e for e in (body.events or []) if e] or ["transcribed", "summarized", "analyzed"]
    if not db.set_key_webhook(kid, body.url.strip(), body.secret.strip(), events):
        raise HTTPException(404, "key not found")
    return {"ok": True}


# ---------------------------------------------------------------- dashboard: recordings
@router.get("/tasks", dependencies=_ADMIN, summary="All recordings + statuses")
def list_tasks():
    items = []
    for j in jobs.list_jobs():
        item = public.task_list_item(j)
        item["callback_result"] = j.get("callback_result")
        item["error"] = j.get("error")
        items.append(item)
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items


@router.get("/tasks/{tid}", dependencies=_ADMIN, summary="One recording (full result)")
def get_task(tid: str):
    job = jobs.get_job(tid)
    if not job:
        raise HTTPException(404, "task not found")
    return public.public_result(job)


@router.get("/tasks/{tid}/runs", dependencies=_ADMIN,
            summary="Version history: every transcribe/summary/analytics run for this recording")
def task_runs(tid: str):
    return db.runs_for(tid)


@router.get("/runs/{run_id}", dependencies=_ADMIN,
            summary="One historical run: params + the full result snapshot as it was produced")
def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@router.post("/tasks/{tid}/rerun", dependencies=_ADMIN,
             summary="Re-run a stage (stage=summary|analytics|transcribe)")
def rerun_task(tid: str, stage: str = "analytics"):
    if not jobs.get_job(tid):
        raise HTTPException(404, "task not found")
    stage = stage.lower()
    if stage == "transcribe":
        ok = jobs.retranscribe(tid)
    elif stage in ("summary", "analytics"):
        ok = jobs.trigger_stage(tid, "summary" if stage == "summary" else "analytics")
    else:
        raise HTTPException(400, "stage must be summary, analytics, or transcribe")
    if not ok:
        raise HTTPException(409, "task not ready to re-run that stage")
    return {"ok": True, "stage": stage}


@router.delete("/tasks/{tid}", dependencies=_ADMIN, summary="Delete a recording + its files")
def delete_task(tid: str):
    if not jobs.delete_job(tid):
        raise HTTPException(404, "task not found")
    return {"ok": True}


# ---------------------------------------------------------------- dashboard: queue + webhooks log
@router.get("/queue", dependencies=_ADMIN, summary="Queue / worker status")
def queue():
    return jobs.queue_status()


@router.get("/webhooks", dependencies=_ADMIN, summary="Webhook delivery log")
def webhook_log(task_id: str | None = None, limit: int = 200):
    return db.list_webhook_events(limit=limit, task_id=task_id)


@router.post("/webhooks/{eid}/redeliver", dependencies=_ADMIN, summary="Re-fire a webhook event")
def redeliver(eid: str):
    if not webhooks.redeliver(eid):
        raise HTTPException(404, "event not found")
    return {"ok": True}
