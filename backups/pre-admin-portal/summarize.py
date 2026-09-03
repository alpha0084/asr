"""Stage 5 — conversation summary + speaker-role inference + sentiment (one LLM call)."""
from .. import llm
from ..utils import transcript_text

SYSTEM = (
    "You are a call-center QA analyst. You analyze call transcripts precisely and "
    "return only what is asked. The transcript may be in any language; analyze it "
    "in its original language but write your output in English."
)

_PROMPT = """Here is a call transcript (speaker-labeled, timestamps in [mm:ss]):

{transcript}

Analyze it and return JSON with EXACTLY these keys:
{{
  "roles": {{"agent": "<the speaker label that is the agent/representative>",
             "customer": "<the speaker label that is the customer>"}},
  "summary": "<3-5 sentence executive summary>",
  "key_points": ["<bullet>", "..."],
  "action_items": ["<follow-up / commitment made>", "..."],
  "outcome": "<one of: Resolved | Escalated | Follow-up | Unresolved>",
  "customer_sentiment": {{"overall": "<positive|neutral|negative>",
                          "trend": "<how it changed across the call>",
                          "frustration_points": ["<moment + why>", "..."]}},
  "agent_tone": "<short description of the agent's tone>"
}}
Base every field strictly on the transcript. Use the exact speaker labels shown."""


def summarize(turns: list) -> dict:
    if not turns:
        return {}
    prompt = _PROMPT.format(transcript=transcript_text(turns))
    return llm.chat_json(prompt, system=SYSTEM)


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
    out = llm.chat_json(prompt, system=SYSTEM)
    return {"agent": out.get("agent"), "customer": out.get("customer")}
