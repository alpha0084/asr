"""Runtime-editable settings, persisted in SQLite, overlaying the .env defaults.

`config.py` holds the immutable boot defaults (from .env). Anything the admin can
change at runtime lives here: it reads the DB value if the admin has set one,
otherwise falls back to the config default. Storing JSON blobs per key keeps the
schema flat — the structured editing happens in the admin UI.
"""
from . import config, db

# setting keys
K_MODELS = "models"
K_SCORECARD = "scorecard"
K_SUMMARY = "summary_fields"
K_WEBHOOK = "webhook"
K_ADMIN = "admin_password_hash"

# The default QA scorecard: categories -> checkpoints, each with a scoring weight.
# This is the single source of truth for the default; the admin can override it
# (analyze.py reads scorecard() at run time). Weights default to 1.0 (equal).
DEFAULT_SCORECARD = [
    {"category": "Opening & Professionalism", "weight": 1.0, "checkpoints": [
        {"name": "Greeting & self-identification", "weight": 1.0},
        {"name": "Professional tone", "weight": 1.0},
        {"name": "Active listening", "weight": 1.0}]},
    {"category": "Communication Skills", "weight": 1.0, "checkpoints": [
        {"name": "Clarity & articulation", "weight": 1.0},
        {"name": "Empathy & acknowledgment", "weight": 1.0},
        {"name": "Interruptions", "weight": 1.0},
        {"name": "Dead air / hold management", "weight": 1.0}]},
    {"category": "Problem Handling", "weight": 1.0, "checkpoints": [
        {"name": "Issue identification", "weight": 1.0},
        {"name": "Knowledge & accuracy", "weight": 1.0},
        {"name": "Resolution", "weight": 1.0},
        {"name": "Effort / efficiency", "weight": 1.0}]},
    {"category": "Compliance", "weight": 1.0, "checkpoints": [
        {"name": "Mandatory disclosures", "weight": 1.0},
        {"name": "Verification", "weight": 1.0},
        {"name": "Data privacy", "weight": 1.0}]},
    {"category": "Closing", "weight": 1.0, "checkpoints": [
        {"name": "Summary & next steps", "weight": 1.0},
        {"name": "Additional help offered", "weight": 1.0},
        {"name": "Proper close", "weight": 1.0}]},
]

# What the summary stage extracts. `outcome_options` drives the allowed outcomes;
# `custom_fields` are extra keys the admin wants pulled from every transcript.
DEFAULT_SUMMARY = {
    "outcome_options": ["Resolved", "Escalated", "Follow-up", "Unresolved"],
    "sentiment": True,
    "custom_fields": [],   # e.g. [{"key": "product_mentioned", "prompt": "Any product named?"}]
}

DEFAULT_WEBHOOK = {
    "enabled": False,
    "url": "",
    "secret": "",
    "events": ["transcribed", "summarized", "analyzed"],
}


# ---------------------------------------------------------------- models / backend
def models() -> dict:
    saved = db.get_setting(K_MODELS) or {}
    return {
        "whisper_model": saved.get("whisper_model") or config.WHISPER_MODEL,
        "asr_backend": saved.get("asr_backend") or config.ASR_BACKEND,
        "ollama_model": saved.get("ollama_model") or config.OLLAMA_MODEL,
        "target_language": saved.get("target_language") or "",
    }


def whisper_model() -> str:
    return models()["whisper_model"]


def asr_backend() -> str:
    return models()["asr_backend"]


def ollama_model() -> str:
    return models()["ollama_model"]


def default_target_language() -> str:
    return models()["target_language"]


# ---------------------------------------------------------------- scorecard / summary
def scorecard() -> list:
    return db.get_setting(K_SCORECARD) or DEFAULT_SCORECARD


def summary_config() -> dict:
    return {**DEFAULT_SUMMARY, **(db.get_setting(K_SUMMARY) or {})}


# ---------------------------------------------------------------- webhook
def webhook() -> dict:
    return {**DEFAULT_WEBHOOK, **(db.get_setting(K_WEBHOOK) or {})}
