"""FastAPI server + fully-documented OpenAPI/Swagger.

Run:   source env.sh && uvicorn backend.server:app --port 8000
UI:    http://localhost:8000
Docs:  http://localhost:8000/docs   (Swagger)   ·   /redoc   ·   /openapi.json
"""
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import APIKeyHeader

from . import admin_api, admin_auth, db, jobs, public, schemas
from .config import API_KEY, DATA_DIR, WHISPER_MODEL

API_DESCRIPTION = """
Local, private call-analytics API — transcription, speaker diarization,
translation, summary, and QA analytics. Everything runs on-device.

## Task flow (Tasks API)

1. **`POST /api/v1/tasks`** — submit a recording (URL). Runs transcribe + diarize
   + translate. Returns a **`task_id`** immediately (status `queued`).
2. **`GET /api/v1/tasks/{id}`** — poll. Watch `status` go `queued → running → done`,
   with `stage` (Transcribing, Diarizing, …) and `detail` (e.g. `2:31 / 5:02`).
   `transcript_ready` flips true when step 3 is available.
3. **`GET /api/v1/tasks/{id}/transcript`** — speaker-wise turns + full text
   (+ translation).
4. **`POST /api/v1/tasks/{id}/summarize`** then poll **`GET …/summary`** — summary,
   outcome, sentiment, action items. *(Async: watch `summary_status`.)*
5. **`POST /api/v1/tasks/{id}/analyze`** then poll **`GET …/analytics`** — QA
   scorecard (0–100, per-checkpoint evidence). *(Async: watch `analytics_status`.)*
6. **`GET /api/v1/tasks`** — list every task (processing + processed).

Summary and analytics run **on demand** so you don't spend LLM time on every call.
Tasks and results are persisted (SQLite) and survive server restarts.

## Authentication

Every **Tasks API** endpoint requires the header **`X-API-Key: <your key>`**
(the value of `API_KEY` in `.env`). Click **Authorize** above and paste your key to
try the endpoints here. The Web-UI endpoints (`/api/jobs*`) are local and unauthenticated.

## Status values

`queued` · `running` · `done` · `error` · `interrupted` (server restarted mid-run).
Stage sub-statuses `summary_status` / `analytics_status`: `null` · `queued` ·
`running` · `done` · `error`.
"""

tags_metadata = [
    {"name": "Tasks API", "description": "Third-party API. **Requires `X-API-Key`.** "
     "Submit → poll → transcript → summarize → analyze. Persistent + async."},
    {"name": "Web UI", "description": "Endpoints used by the local browser UI. No auth."},
    {"name": "Legacy", "description": "Deprecated all-in-one endpoint — use the Tasks API instead."},
]

app = FastAPI(
    title="ASR Call Analytics API",
    description=API_DESCRIPTION,
    version="0.5.0",
    openapi_tags=tags_metadata,
    contact={"name": "ASR Call Analytics"},
)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
UPLOAD_DIR = DATA_DIR / "_uploads"

# --- API-key security scheme (shows the "Authorize" button in Swagger) ---
api_key_scheme = APIKeyHeader(
    name="X-API-Key", auto_error=False,
    description="Your API key (value of `API_KEY` in `.env`). Required on all Tasks API calls.")


def require_api_key(key: str | None = Security(api_key_scheme)):
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key header")
    return key


_AUTH = [Depends(require_api_key)]
_ERRORS = {
    401: {"model": schemas.ErrorResponse, "description": "Missing or invalid X-API-Key"},
    404: {"model": schemas.ErrorResponse, "description": "Task not found"},
}


app.include_router(admin_api.router)


@app.on_event("startup")
def _startup():
    jobs.load_from_db()      # repopulate tasks from SQLite
    admin_auth.bootstrap()   # seed admin password from ADMIN_PASSWORD on first run


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_page():
    return (FRONTEND / "admin.html").read_text()


def _require(tid: str) -> dict:
    job = jobs.get_job(tid)
    if not job:
        raise HTTPException(404, "task not found")
    return job


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return (FRONTEND / "index.html").read_text()


# ============================================================================
# TASKS API  (third-party, API-key protected)
# ============================================================================
@app.post("/api/v1/tasks", status_code=202, tags=["Tasks API"], dependencies=_AUTH,
          response_model=schemas.CreateTaskResponse, responses=_ERRORS,
          summary="Enqueue a recording",
          response_description="Task accepted; poll `status_url`.")
def create_task(req: schemas.TranscribeRequest):
    """Queue a recording for **transcribe + diarize + translate**.

    Summary and QA analytics are **not** run here — call `/summarize` and
    `/analyze` afterward. Returns a `task_id` used for every follow-up call.
    """
    model = "large-v3" if req.engine.strip().lower() in ("auto", "") else req.engine.strip()
    language = None if req.language.strip().lower() in ("auto", "") else req.language.strip()
    target = "English" if req.translate_to_english else None
    name = req.audio_path.split("?")[0].rsplit("/", 1)[-1] or "audio"

    tid = jobs.create_job(
        req.audio_path, name, model=model, language=language, speakers=req.speakers,
        target_language=target, include_summary=False, include_analytics=False,
        callback_url=req.callback_url)
    return {"task_id": tid, "status": "queued", "status_url": f"/api/v1/tasks/{tid}",
            "message": "queued"}


@app.get("/api/v1/tasks", tags=["Tasks API"], dependencies=_AUTH,
         response_model=list[schemas.TaskListItem], responses=_ERRORS,
         summary="List all tasks",
         response_description="All tasks (processing + processed), newest first.")
def list_tasks():
    """List every task with its status and per-stage progress."""
    items = [public.task_list_item(j) for j in jobs.list_jobs()]
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items


@app.get("/api/v1/tasks/{tid}", tags=["Tasks API"], dependencies=_AUTH,
         response_model=schemas.TaskStatus, responses=_ERRORS,
         summary="Poll task status / progress")
def get_task(tid: str):
    """Current `status`, `stage`, live `detail`, and `*_ready` flags for each result."""
    return public.status_result(_require(tid))


@app.get("/api/v1/tasks/{tid}/transcript", tags=["Tasks API"], dependencies=_AUTH,
         response_model=schemas.TranscriptResponse, responses=_ERRORS,
         summary="Get speaker-wise transcript + full text")
def task_transcript(tid: str):
    """Speaker-labeled turns (with roles once known), full text, and translation."""
    return public.transcript_result(_require(tid))


@app.post("/api/v1/tasks/{tid}/summarize", status_code=202, tags=["Tasks API"],
          dependencies=_AUTH, response_model=schemas.StageQueued,
          responses={**_ERRORS, 409: {"model": schemas.ErrorResponse,
                     "description": "Transcription not finished yet"}},
          summary="Run the summary (async)",
          response_description="Queued; poll `GET /summary` for `summary_status: done`.")
def task_summarize(tid: str):
    """Kick off the conversation summary. It runs in the background; poll
    `GET /api/v1/tasks/{id}/summary` until `summary_status` is `done`."""
    if not jobs.trigger_stage(tid, "summary"):
        raise HTTPException(409, "task not ready (transcription must finish first)")
    return {"task_id": tid, "summary_status": "queued",
            "result_url": f"/api/v1/tasks/{tid}/summary"}


@app.get("/api/v1/tasks/{tid}/summary", tags=["Tasks API"], dependencies=_AUTH,
         response_model=schemas.SummaryResponse, responses=_ERRORS,
         summary="Get the summary result")
def task_summary(tid: str):
    """Summary, outcome, customer sentiment, agent tone, key points, action items."""
    return public.summary_result(_require(tid))


@app.post("/api/v1/tasks/{tid}/analyze", status_code=202, tags=["Tasks API"],
          dependencies=_AUTH, response_model=schemas.StageQueued,
          responses={**_ERRORS, 409: {"model": schemas.ErrorResponse,
                     "description": "Transcription not finished yet"}},
          summary="Run the QA analytics (async)",
          response_description="Queued; poll `GET /analytics` for `analytics_status: done`.")
def task_analyze(tid: str):
    """Kick off the QA scorecard (5 checkpoint categories). Runs in the background;
    poll `GET /api/v1/tasks/{id}/analytics` until `analytics_status` is `done`."""
    if not jobs.trigger_stage(tid, "analytics"):
        raise HTTPException(409, "task not ready (transcription must finish first)")
    return {"task_id": tid, "analytics_status": "queued",
            "result_url": f"/api/v1/tasks/{tid}/analytics"}


@app.get("/api/v1/tasks/{tid}/analytics", tags=["Tasks API"], dependencies=_AUTH,
         response_model=schemas.AnalyticsResponse, responses=_ERRORS,
         summary="Get the QA analytics result")
def task_analytics(tid: str):
    """0–100 score + full scorecard (checkpoint, score, verdict, evidence, suggestion)."""
    return public.analytics_result(_require(tid))


@app.get("/api/v1/tasks/{tid}/result", tags=["Tasks API"], dependencies=_AUTH,
         responses=_ERRORS, summary="Get everything available so far")
def task_full_result(tid: str):
    """Combined payload: metadata + speaker-wise + full text + summary + analytics
    (whichever have been run)."""
    return public.public_result(_require(tid))


# ============================================================================
# WEB UI  (local browser; no auth)
# ============================================================================
@app.post("/api/jobs", tags=["Web UI"], summary="Web submit (file or URL)")
async def create_job(
    file: UploadFile = File(None),
    url: str = Form(""),
    model: str = Form(WHISPER_MODEL),
    language: str = Form(""),
    speakers: str = Form(""),
    target_language: str = Form(""),
):
    """Browser upload/URL submit — transcribe + diarize + translate only."""
    language = language.strip() or None
    speakers_int = int(speakers) if speakers.strip().isdigit() else None

    if file is not None and file.filename:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        src, name = str(dest), file.filename
    elif url.strip():
        u = url.strip()
        name = u.split("?")[0].rsplit("/", 1)[-1] or "url-audio"
        src = u
    else:
        raise HTTPException(400, "Provide an audio file or a URL.")

    jid = jobs.create_job(src, name, model=model, language=language, speakers=speakers_int,
                          target_language=target_language,
                          include_summary=False, include_analytics=False)
    return {"id": jid}


@app.post("/api/jobs/{jid}/summarize", tags=["Web UI"], summary="Web: run summary")
def summarize_job(jid: str):
    if not jobs.trigger_stage(jid, "summary"):
        raise HTTPException(409, "job not ready (transcribe must finish first)")
    return {"ok": True, "stage": "summary"}


@app.post("/api/jobs/{jid}/analyze", tags=["Web UI"], summary="Web: run analytics")
def analyze_job(jid: str):
    if not jobs.trigger_stage(jid, "analytics"):
        raise HTTPException(409, "job not ready (transcribe must finish first)")
    return {"ok": True, "stage": "analytics"}


@app.get("/api/jobs", tags=["Web UI"], summary="Web: list jobs")
def all_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{jid}", tags=["Web UI"], summary="Web: job status")
def one_job(jid: str):
    job = jobs.get_job(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/jobs/{jid}/audio", tags=["Web UI"], summary="Web: stream the recording")
def job_audio(jid: str):
    """Serve the original recording for in-browser playback."""
    job = jobs.get_job(jid)
    path = (job.get("result") or {}).get("input") if job else None
    if not path or not os.path.exists(path):
        raise HTTPException(404, "audio not found")
    return FileResponse(path)


# ============================================================================
# LEGACY  (all-in-one; kept for backward compatibility)
# ============================================================================
@app.post("/api/v1/jobs", status_code=202, tags=["Legacy"], dependencies=_AUTH,
          responses=_ERRORS, summary="All-in-one (deprecated)",
          response_description="Runs transcribe + summary + QA in one job.")
def create_v1_job(req: schemas.TranscribeRequest):
    """Deprecated. Runs everything in a single job (webhook or poll). New
    integrations should use the **Tasks API** instead."""
    model = "large-v3" if req.engine.strip().lower() in ("auto", "") else req.engine.strip()
    language = None if req.language.strip().lower() in ("auto", "") else req.language.strip()
    target = "English" if req.translate_to_english else None
    name = req.audio_path.split("?")[0].rsplit("/", 1)[-1] or "audio"

    jid = jobs.create_job(
        req.audio_path, name, model=model, language=language, speakers=req.speakers,
        target_language=target, include_summary=True, include_analytics=True,
        callback_url=req.callback_url)
    return {"job_id": jid, "status": "queued", "status_url": f"/api/v1/jobs/{jid}",
            "message": ("Processing started. Results will be POSTed to callback_url when ready."
                        if req.callback_url else "Processing started. Poll status_url.")}


@app.get("/api/v1/jobs/{jid}", tags=["Legacy"], dependencies=_AUTH, responses=_ERRORS,
         summary="All-in-one status/result (deprecated)")
def get_v1_job(jid: str):
    job = jobs.get_job(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return public.public_result(job)
