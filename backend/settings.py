"""Runtime-editable settings, persisted in SQLite, overlaying the .env defaults.

`config.py` holds the immutable boot defaults (from .env). Anything the admin can
change at runtime lives here: it reads the DB value if the admin has set one,
otherwise falls back to the config default. Storing JSON blobs per key keeps the
schema flat — the structured editing happens in the admin UI.
"""
import uuid

from . import config, db

# setting keys
K_MODELS = "models"
K_SCORECARD = "scorecard"
K_SCORECARD_VERSION = "scorecard_version"
K_SUMMARY = "summary_fields"
K_WEBHOOK = "webhook"
K_ADMIN = "admin_password_hash"

# The default QA scorecard: categories -> checkpoints, each with a scoring weight.
# This is the single source of truth for the default; the admin can override it
# (analyze.py reads scorecard() at run time). Weights default to 1.0 (equal).
DEFAULT_SCORECARD = [
    {"id": "opening", "category": "Opening & Professionalism", "weight": 1.0, "checkpoints": [
        {"id": "greeting_self_id", "name": "Greeting & self-identification", "weight": 1.0},
        {"id": "professional_tone", "name": "Professional tone", "weight": 1.0},
        {"id": "active_listening", "name": "Active listening", "weight": 1.0}]},
    {"id": "communication", "category": "Communication Skills", "weight": 1.0, "checkpoints": [
        {"id": "clarity", "name": "Clarity & articulation", "weight": 1.0},
        {"id": "empathy", "name": "Empathy & acknowledgment", "weight": 1.0},
        {"id": "interruptions", "name": "Interruptions", "weight": 1.0},
        {"id": "dead_air", "name": "Dead air / hold management", "weight": 1.0}]},
    {"id": "problem_handling", "category": "Problem Handling", "weight": 1.0, "checkpoints": [
        {"id": "issue_id", "name": "Issue identification", "weight": 1.0},
        {"id": "knowledge", "name": "Knowledge & accuracy", "weight": 1.0},
        {"id": "resolution", "name": "Resolution", "weight": 1.0},
        {"id": "effort", "name": "Effort / efficiency", "weight": 1.0}]},
    {"id": "compliance", "category": "Compliance", "weight": 1.0, "checkpoints": [
        {"id": "disclosures", "name": "Mandatory disclosures", "weight": 1.0},
        {"id": "verification", "name": "Verification", "weight": 1.0},
        {"id": "data_privacy", "name": "Data privacy", "weight": 1.0}]},
    {"id": "closing", "category": "Closing", "weight": 1.0, "checkpoints": [
        {"id": "summary_next_steps", "name": "Summary & next steps", "weight": 1.0},
        {"id": "additional_help", "name": "Additional help offered", "weight": 1.0},
        {"id": "proper_close", "name": "Proper close", "weight": 1.0}]},
]

# What the summary stage extracts. `outcome_options` drives the allowed outcomes;
# `custom_fields` are extra keys the admin wants pulled from every transcript.
DEFAULT_SUMMARY = {
    "outcome_options": ["Resolved", "Escalated", "Follow-up", "Unresolved"],
    "sentiment": True,
    "custom_fields": [],   # e.g. [{"key": "product_mentioned", "prompt": "Any product named?"}]
}

DEFAULT_WEBHOOK = {
    "enabled": config.WEBHOOK_ENABLED,
    "url": config.WEBHOOK_URL,
    "secret": config.WEBHOOK_SECRET,
    "events": ["transcribed", "summarized", "analyzed"],
}


# ---------------------------------------------------------------- models / backend
def models() -> dict:
    """Per-process model selection (admin-editable), falling back to .env defaults.

    Each pipeline stage has its own model so the admin can trade speed vs quality
    independently: transcription (Whisper), diarization / summary / analytics (LLM).
    """
    saved = db.get_setting(K_MODELS) or {}
    default_llm = saved.get("ollama_model") or config.OLLAMA_MODEL
    return {
        "transcribe_model": saved.get("transcribe_model") or saved.get("whisper_model") or config.WHISPER_MODEL,
        "asr_backend": saved.get("asr_backend") or config.ASR_BACKEND,
        "speaker_mode": saved.get("speaker_mode") or config.SPEAKER_MODE,
        "diarize_model": saved.get("diarize_model") or default_llm,
        "summary_model": saved.get("summary_model") or default_llm,
        "analytics_model": saved.get("analytics_model") or default_llm,
        "target_language": saved.get("target_language") or "",
    }


def whisper_model() -> str:
    return models()["transcribe_model"]


def asr_backend() -> str:
    return models()["asr_backend"]


def speaker_mode() -> str:
    return models()["speaker_mode"]


def diarize_model() -> str:
    return models()["diarize_model"]


def summary_model() -> str:
    return models()["summary_model"]


def analytics_model() -> str:
    return models()["analytics_model"]


def ollama_model() -> str:
    # Backwards-compatible general default (used where a stage isn't specified).
    return models()["summary_model"]


def default_target_language() -> str:
    return models()["target_language"]


# ---------------------------------------------------------------- scorecard / summary
def _ensure_ids(sc: list):
    """Assign a stable id to any category/checkpoint missing one."""
    changed = False
    for c in sc or []:
        if not c.get("id"):
            c["id"] = "cat_" + uuid.uuid4().hex[:8]
            changed = True
        for cp in c.get("checkpoints", []):
            if not cp.get("id"):
                cp["id"] = "cp_" + uuid.uuid4().hex[:8]
                changed = True
    return sc, changed


def scorecard() -> list:
    """Categories/checkpoints with STABLE ids. Backfills ids on legacy configs so
    rollups downstream can key on ids that survive checkpoint renames."""
    sc = db.get_setting(K_SCORECARD)
    if sc is None:
        return DEFAULT_SCORECARD          # defaults already carry stable ids
    sc, changed = _ensure_ids(sc)
    if changed:
        db.set_setting(K_SCORECARD, sc)
    return sc


def scorecard_version() -> int:
    return int(db.get_setting(K_SCORECARD_VERSION) or 1)


def summary_config() -> dict:
    return {**DEFAULT_SUMMARY, **(db.get_setting(K_SUMMARY) or {})}


# ---------------------------------------------------------------- webhook
def webhook() -> dict:
    return {**DEFAULT_WEBHOOK, **(db.get_setting(K_WEBHOOK) or {})}
