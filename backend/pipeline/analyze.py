"""Stage 6 — QA scorecard, scored with evidence.

The scorecard (categories, checkpoints, weights) is admin-configurable — see
`settings.scorecard()`; the default lives in `settings.DEFAULT_SCORECARD`.
"""
import re

from .. import llm, settings
from ..utils import transcript_text

# Constrain the model's free-text verdict to a fixed enum so downstream $inc
# counters (met-rate, N/A-rate, …) are unambiguous.
_VERDICT_MAP = {
    "met": "Met", "partiallymet": "Partially Met", "partial": "Partially Met",
    "notmet": "Not Met", "fail": "Not Met", "failed": "Not Met",
    "na": "N/A", "notapplicable": "N/A", "n/a": "N/A",
}


def _norm_verdict(v) -> str:
    key = re.sub(r"[^a-z/]", "", str(v or "").lower())
    return _VERDICT_MAP.get(key, "N/A")

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
    out = llm.chat_json(prompt, system=SYSTEM, model=settings.analytics_model())
    rows = out.get("items", []) if isinstance(out, dict) else []
    for r in rows:
        r["category"] = category
    return rows


def _overall(rows: list, weights: dict) -> dict:
    """Weighted 0-100 rollup; compliance 'Not Met' is a heavy penalty.

    `weights` maps (category, checkpoint) -> weight (category weight × checkpoint
    weight). Defaults are all 1.0, so an unweighted scorecard behaves as before.
    """
    scored = [r for r in rows if isinstance(r.get("score"), (int, float)) and r["score"] > 0]
    num = sum(r["score"] * weights.get((r.get("category"), r.get("checkpoint")), 1.0) for r in scored)
    den = sum(5 * weights.get((r.get("category"), r.get("checkpoint")), 1.0) for r in scored)
    base = round(num / den * 100) if den else 0

    # The compliance cap applies to a category literally named "Compliance".
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
    scorecard = settings.scorecard()
    rows, weights = [], {}
    for idx, cat in enumerate(scorecard, 1):
        category = cat["category"]
        cat_w = float(cat.get("weight", 1.0) or 1.0)
        checkpoints = cat.get("checkpoints", [])
        for cp in checkpoints:
            weights[(category, cp["name"])] = cat_w * float(cp.get("weight", 1.0) or 1.0)
        if on_progress:
            on_progress(idx, len(scorecard), category)
        rows.extend(_score_category(category, [cp["name"] for cp in checkpoints], ctx))

    # Attach STABLE ids + a constrained verdict enum + met/applicable flags so the
    # webhook/DB carry keys that survive checkpoint renames, and rollups are exact.
    cat_ids = {c["category"]: c.get("id") for c in scorecard}
    cp_ids = {(c["category"], cp["name"]): cp.get("id")
              for c in scorecard for cp in c.get("checkpoints", [])}
    for r in rows:
        r["verdict"] = _norm_verdict(r.get("verdict"))
        r["met"] = 1 if r["verdict"] == "Met" else 0        # int flag (met-rate)
        r["applicable"] = r["verdict"] != "N/A"             # bool flag (N/A-rate)
        r["category_id"] = cat_ids.get(r.get("category"))
        r["checkpoint_id"] = cp_ids.get((r.get("category"), r.get("checkpoint")))

    return {"scorecard": rows, "scorecard_version": settings.scorecard_version(),
            **_overall(rows, weights)}
