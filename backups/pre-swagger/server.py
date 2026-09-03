"""FastAPI server — upload a file or paste a URL, get a speaker-wise transcript.

Run:  source env.sh && uvicorn backend.server:app --port 8000
Open: http://localhost:8000
"""
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import AliasChoices, BaseModel, Field

from . import jobs, public
from .config import API_KEY, DATA_DIR, WHISPER_MODEL

app = FastAPI(title="ASR Call Analytics")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
UPLOAD_DIR = DATA_DIR / "_uploads"


@app.on_event("startup")
def _startup():
    jobs.load_from_db()      # repopulate tasks from SQLite


def require_api_key(x_api_key: str = Header(None)):
    """Auth for the public /api/v1 task API. No-op if API_KEY is unset (dev)."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key header")


@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND / "index.html").read_text()


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(None),
    url: str = Form(""),
    model: str = Form(WHISPER_MODEL),
    language: str = Form(""),
    speakers: str = Form(""),
    target_language: str = Form(""),
):
    """Web submit: transcribe + diarize + translate only. Summary/QA run on demand."""
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


@app.post("/api/jobs/{jid}/summarize")
def summarize_job(jid: str):
    if not jobs.trigger_stage(jid, "summary"):
        raise HTTPException(409, "job not ready (transcribe must finish first)")
    return {"ok": True, "stage": "summary"}


@app.post("/api/jobs/{jid}/analyze")
def analyze_job(jid: str):
    if not jobs.trigger_stage(jid, "analytics"):
        raise HTTPException(409, "job not ready (transcribe must finish first)")
    return {"ok": True, "stage": "analytics"}


@app.get("/api/jobs")
def all_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{jid}")
def one_job(jid: str):
    job = jobs.get_job(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# ----------------------------------------------------------------------------
# Public v1 API — for third-party integrations (async + webhook callback)
# ----------------------------------------------------------------------------
class TranscribeRequest(BaseModel):
    model_config = {"populate_by_name": True, "extra": "ignore"}

    audio_path: str = Field(validation_alias=AliasChoices(
        "Audio Path", "audio_path", "audio_url", "AudioPath", "url"))
    callback_url: str | None = None
    engine: str = "auto"            # "auto" -> large-v3 (best); or a whisper size
    language: str = "auto"          # "auto" -> auto-detect; or a code like "hi"
    translate_to_english: bool = False
    speakers: int | None = None     # optional: pin speaker count (e.g. 2)


@app.post("/api/v1/jobs", status_code=202)
def create_v1_job(req: TranscribeRequest):
    model = "large-v3" if req.engine.strip().lower() in ("auto", "") else req.engine.strip()
    language = None if req.language.strip().lower() in ("auto", "") else req.language.strip()
    target = "English" if req.translate_to_english else None
    name = req.audio_path.split("?")[0].rsplit("/", 1)[-1] or "audio"

    jid = jobs.create_job(
        req.audio_path, name, model=model, language=language, speakers=req.speakers,
        target_language=target, include_summary=True, include_analytics=True,
        callback_url=req.callback_url,
    )
    return {
        "job_id": jid,
        "status": "queued",
        "status_url": f"/api/v1/jobs/{jid}",
        "message": ("Processing started. Results will be POSTed to callback_url when ready."
                    if req.callback_url else
                    "Processing started. Poll status_url for the result."),
    }


@app.get("/api/v1/jobs/{jid}")
def get_v1_job(jid: str):
    job = jobs.get_job(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return JSONResponse(public.public_result(job))


# ----------------------------------------------------------------------------
# Public v1 TASK API — two-step, persistent, API-key protected.
# Submit → poll → then transcript / summarize / analyze separately.
# All endpoints require the X-API-Key header.
# ----------------------------------------------------------------------------
_AUTH = [Depends(require_api_key)]


def _require(tid: str) -> dict:
    job = jobs.get_job(tid)
    if not job:
        raise HTTPException(404, "task not found")
    return job


@app.post("/api/v1/tasks", status_code=202, dependencies=_AUTH)
def create_task(req: TranscribeRequest):
    """Enqueue a recording (transcribe + diarize + translate). Returns a task_id."""
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


@app.get("/api/v1/tasks", dependencies=_AUTH)
def list_tasks():
    """All tasks (processing + processed), newest first."""
    items = [public.task_list_item(j) for j in jobs.list_jobs()]
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items


@app.get("/api/v1/tasks/{tid}", dependencies=_AUTH)
def get_task(tid: str):
    """Poll: status, stage, progress detail, and which stages are ready."""
    return public.status_result(_require(tid))


@app.get("/api/v1/tasks/{tid}/transcript", dependencies=_AUTH)
def task_transcript(tid: str):
    """Speaker-wise turns + full text (+ translation)."""
    return public.transcript_result(_require(tid))


@app.post("/api/v1/tasks/{tid}/summarize", status_code=202, dependencies=_AUTH)
def task_summarize(tid: str):
    if not jobs.trigger_stage(tid, "summary"):
        raise HTTPException(409, "task not ready (transcription must finish first)")
    return {"task_id": tid, "summary_status": "queued",
            "result_url": f"/api/v1/tasks/{tid}/summary"}


@app.get("/api/v1/tasks/{tid}/summary", dependencies=_AUTH)
def task_summary(tid: str):
    return public.summary_result(_require(tid))


@app.post("/api/v1/tasks/{tid}/analyze", status_code=202, dependencies=_AUTH)
def task_analyze(tid: str):
    if not jobs.trigger_stage(tid, "analytics"):
        raise HTTPException(409, "task not ready (transcription must finish first)")
    return {"task_id": tid, "analytics_status": "queued",
            "result_url": f"/api/v1/tasks/{tid}/analytics"}


@app.get("/api/v1/tasks/{tid}/analytics", dependencies=_AUTH)
def task_analytics(tid: str):
    return public.analytics_result(_require(tid))


@app.get("/api/v1/tasks/{tid}/result", dependencies=_AUTH)
def task_full_result(tid: str):
    """Everything available so far in one payload."""
    return JSONResponse(public.public_result(_require(tid)))
