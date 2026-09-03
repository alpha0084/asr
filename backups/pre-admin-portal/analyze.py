"""Stage 6 — QA scorecard (the standard checkpoint set), scored with evidence."""
from .. import llm
from ..utils import transcript_text

# The standard scorecard, grouped by category.
SCORECARD = [
    ("Opening & Professionalism", [
        "Greeting & self-identification", "Professional tone", "Active listening"]),
    ("Communication Skills", [
        "Clarity & articulation", "Empathy & acknowledgment",
        "Interruptions", "Dead air / hold management"]),
    ("Problem Handling", [
        "Issue identification", "Knowledge & accuracy",
        "Resolution", "Effort / efficiency"]),
    ("Compliance", [
        "Mandatory disclosures", "Verification", "Data privacy"]),
    ("Closing", [
        "Summary & next steps", "Additional help offered", "Proper close"]),
]

SYSTEM = (
    "You are a strict but fair call-center QA evaluator. Score only the AGENT's "
    "performance, using evidence from the transcript. The transcript may be in any "
    "language; write your output in English. Return only valid JSON."
)

_PROMPT = """Call transcript (the agent is {agent}, the customer is {customer}):

{transcript}

Score the AGENT on these checkpoints for the category "{category}":
{items}

Return JSON: {{"items": [
  {{"checkpoint": "<exact name>",
    "score": <integer 1-5, or 0 if not applicable>,
    "verdict": "<Met | Partially Met | Not Met | N/A>",
    "evidence": "<short quote + [mm:ss] from the transcript, or 'no evidence'>",
    "suggestion": "<one concrete improvement, or ''>"}}
]}}
Rules: quote real evidence with its timestamp. If a checkpoint doesn't apply to this
call, use verdict "N/A" and score 0. Be consistent and concise."""


def _score_category(category: str, items: list, ctx: dict) -> list:
    prompt = _PROMPT.format(
        category=category,
        items="\n".join(f"- {i}" for i in items),
        transcript=ctx["transcript"],
        agent=ctx["agent"], customer=ctx["customer"],
    )
    out = llm.chat_json(prompt, system=SYSTEM)
    rows = out.get("items", []) if isinstance(out, dict) else []
    for r in rows:
        r["category"] = category
    return rows


def _overall(rows: list) -> dict:
    """Weighted 0-100 rollup; compliance 'Not Met' is a heavy penalty."""
    scored = [r for r in rows if isinstance(r.get("score"), (int, float)) and r["score"] > 0]
    base = round(sum(r["score"] for r in scored) / (len(scored) or 1) / 5 * 100)

    compliance_fail = any(
        r.get("category") == "Compliance" and r.get("verdict") == "Not Met" for r in rows
    )
    overall = min(base, 50) if compliance_fail else base
    return {
        "overall_score": overall,
        "raw_score": base,
        "compliance_fail": compliance_fail,
        "checkpoints_scored": len(scored),
    }


def analyze(turns: list, roles: dict | None = None, on_progress=None) -> dict:
    if not turns:
        return {}
    roles = roles or {}
    ctx = {
        "transcript": transcript_text(turns),
        "agent": roles.get("agent", "the agent"),
        "customer": roles.get("customer", "the customer"),
    }
    rows = []
    for idx, (category, items) in enumerate(SCORECARD, 1):
        if on_progress:
            on_progress(idx, len(SCORECARD), category)
        rows.extend(_score_category(category, items, ctx))
    return {"scorecard": rows, **_overall(rows)}
