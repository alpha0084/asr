"""Stage 5 — conversation summary + speaker-role inference + sentiment (one LLM call).

The extracted fields (outcome options, whether to score sentiment, and any custom
fields) are admin-configurable — see `settings.summary_config()`.
"""
from .. import llm, settings
from ..utils import transcript_text

SYSTEM = (
    "You are a call-center QA analyst. You analyze call transcripts precisely and "
    "return only what is asked. The transcript may be in any language; analyze it "
    "in its original language but write your output in English."
)


def _build_prompt(transcript: str, cfg: dict) -> str:
    """Assemble the requested-JSON-keys prompt from the admin's summary config."""
    outcomes = " | ".join(cfg.get("outcome_options") or ["Resolved", "Escalated", "Follow-up", "Unresolved"])
    keys = [
        '"roles": {"agent": "<the speaker label that is the agent/representative>", "customer": "<the speaker label that is the customer>"}',
        '"summary": "<3-5 sentence executive summary>"',
        '"key_points": ["<bullet>", "..."]',
        '"action_items": ["<follow-up / commitment made>", "..."]',
        f'"outcome": "<one of: {outcomes}>"',
    ]
    if cfg.get("sentiment", True):
        keys.append('"customer_sentiment": {"overall": "<positive|neutral|negative>", '
                    '"trend": "<how it changed across the call>", "frustration_points": ["<moment + why>", "..."]}')
    keys.append('"agent_tone": "<short description of the agent\'s tone>"')
    for field in cfg.get("custom_fields") or []:
        key = (field.get("key") or "").strip()
        if key:
            keys.append(f'"{key}": "<{field.get("prompt", "")}>"')
    body = ",\n  ".join(keys)
    return ("Here is a call transcript (speaker-labeled, timestamps in [mm:ss]):\n\n"
            f"{transcript}\n\n"
            "Analyze it and return JSON with EXACTLY these keys:\n"
            f"{{\n  {body}\n}}\n"
            "Base every field strictly on the transcript. Use the exact speaker labels shown.")


def summarize(turns: list) -> dict:
    if not turns:
        return {}
    prompt = _build_prompt(transcript_text(turns), settings.summary_config())
    return llm.chat_json(prompt, system=SYSTEM, model=settings.summary_model())


_ROLE_PROMPT = """Here is a call transcript (speaker-labeled):

{transcript}

Identify which speaker label is the agent/representative and which is the customer.
Return JSON exactly: {{"agent": "<speaker label>", "customer": "<speaker label>"}}
Use the exact speaker labels shown."""


def infer_roles(turns: list) -> dict:
    """Cheap one-call role detection (agent vs customer), for Analyze-without-Summarize."""
    if not turns:
        return {}
    prompt = _ROLE_PROMPT.format(transcript=transcript_text(turns))
    out = llm.chat_json(prompt, system=SYSTEM, model=settings.summary_model())
    return {"agent": out.get("agent"), "customer": out.get("customer")}
