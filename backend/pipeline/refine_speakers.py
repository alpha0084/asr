"""LLM-based speaker assignment for mono calls.

Instead of separating voices acoustically (pyannote), we split the transcript
into sentence-level units (anchored to Whisper word timestamps) and let the LLM
label each as AGENT or CUSTOMER using conversational logic — which works even
when the audio is mono/quiet/8kHz and voices can't be told apart acoustically.

Note: role labels can still wobble on garbled transcripts; the split (which fixes
the "merged turns" problem) is the main win. Improving transcription quality
(eval + audio cleanup) lifts this too.
"""
import re

from .. import llm, settings
from ..config import SPEAKER_PAUSE_GAP

_SENT_END = re.compile(r"[.?!।]$")     # includes Devanagari danda
_CHUNK = 400                           # label the whole call in one pass when possible
_OPTS = {"num_ctx": 16384}             # single-pass room for long calls

SYSTEM = (
    "You are an expert call-center analyst. You label who speaks each line of a "
    "two-person OUTBOUND phone call (a company called a customer) whose transcript "
    "has NO speaker labels. There are exactly TWO speakers and each stays the SAME "
    "person for the whole call.\n\n"
    "AGENT — the caller / company rep. They verify who they reached, identify "
    "themselves or their company, state the PURPOSE of the call and DRIVE it. The "
    "purpose can be ANY of: a sales pitch, a support/service follow-up, a "
    "COLLECTIONS or payment reminder (EMI due, credit-card bill, loan, 'last date', "
    "amount owed), an appointment, or an identity/KYC check. They ask the questions "
    "and tell the other party what is due / offered / required.\n"
    "CUSTOMER — the person who was called. They confirm their identity ('yes, "
    "speaking', 'this is <name>'), answer the agent's questions, describe their own "
    "situation, and ask about the amount / due date / how to pay / the product.\n\n"
    "ANCHOR the opening of an OUTBOUND call: the phone is answered by the CUSTOMER, "
    "so the first 'Hello' is the CUSTOMER; the caller then asks 'am I speaking with "
    "<name>?' — that line is the AGENT; the reply 'yes, you are speaking with "
    "<name>' is the CUSTOMER. Fix the two roles from this exchange, then stay "
    "globally consistent — each role is ONE fixed person the whole call.\n\n"
    "DECIDE EACH LINE BY CONTENT AND INTENT FIRST: who is verifying / demanding "
    "payment / asking qualifying questions / driving = AGENT; who is confirming / "
    "answering / describing their own situation = CUSTOMER. A speaker OFTEN says "
    "SEVERAL lines in a row (a greeting then a question; a demand then its detail) — "
    "do NOT force strict agent/customer alternation; use turn-taking only as a "
    "tie-breaker when content is neutral (a question is usually answered by the "
    "OTHER speaker next). Backchannels ('yeah', 'okay', 'hmm', 'ji', 'haan', "
    "'right') belong to whoever is listening at that moment. Humans interrupt, "
    "repeat, and code-switch (Hindi/English) — read the intent, not just keywords.\n"
    "Return ONLY valid JSON."
)

_PROMPT = """Numbered lines of the call (one utterance per line):

{lines}

Label the speaker of EVERY line above. Return JSON exactly:
{{"labels": {{"0": "agent", "1": "customer", ...}}}}
Use only "agent" or "customer". Every line number must appear exactly once."""


_ORIENT_SYSTEM = (
    "You identify the AGENT in a two-person OUTBOUND phone call (a company called a "
    "customer). The AGENT is the caller / company rep: they verify who they reached "
    "('am I speaking with <name>?'), identify themselves or their company, and state "
    "the PURPOSE and DRIVE the call — a sales pitch, a support follow-up, a "
    "COLLECTIONS / payment reminder (EMI, credit-card bill, loan, 'last date', amount "
    "due), an appointment, or a KYC check — asking the questions and stating what is "
    "due / offered / required. The CUSTOMER is the person who was called: they confirm "
    "their identity, answer, and ask about the amount / due date / how to pay. Return "
    "only valid JSON."
)

_ORIENT_PROMPT = """A phone call has two speakers.

SPEAKER A said (sample):
{a}

SPEAKER B said (sample):
{b}

Which speaker is the AGENT — the caller who verifies the other's identity, states the
purpose, and drives the call with questions/demands (sales, collections, support or KYC)?
Return JSON exactly: {{"agent": "A"}} or {{"agent": "B"}}"""


def orient(by_speaker: dict) -> dict:
    """Map pyannote cluster ids -> 'Agent'/'Customer' with ONE global LLM decision.

    by_speaker: {speaker_id: [texts]}. Reliable because the model only decides which
    of two aggregated speakers is the agent — not per-line labeling.
    """
    ranked = sorted(by_speaker.items(), key=lambda kv: -sum(len(t) for t in kv[1]))
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {ranked[0][0]: "Agent"}

    a_id, b_id = ranked[0][0], ranked[1][0]
    a_txt = " ".join(by_speaker[a_id])[:1500]
    b_txt = " ".join(by_speaker[b_id])[:1500]
    try:
        out = llm.chat_json(_ORIENT_PROMPT.format(a=a_txt, b=b_txt),
                            system=_ORIENT_SYSTEM, options=_OPTS,
                            model=settings.diarize_model())
        agent = str(out.get("agent", "A")).strip().upper()
    except Exception:
        agent = "A"                      # fallback: the bigger talker is the agent
    mapping = {a_id: "Agent", b_id: "Customer"} if agent != "B" else {a_id: "Customer", b_id: "Agent"}
    for i, (sid, _) in enumerate(ranked[2:], start=3):   # extra clusters, if any
        mapping[sid] = f"Speaker {i}"
    return mapping


def _mk_unit(words):
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"].strip() for w in words).strip(),
    }


def sentence_units(segments, gap: float | None = None):
    """Split segments into utterance-level units with timestamps (from word times).

    Boundaries are drawn at sentence punctuation OR at a silence gap between
    consecutive words longer than `gap` seconds. The pause rule catches turn
    changes on run-on ASR output that has no punctuation (common at 8kHz mono),
    so two speakers stop landing in one unit. Over-splitting within one speaker
    is harmless — build_turns re-merges consecutive same-speaker units.
    """
    gap = SPEAKER_PAUSE_GAP if gap is None else gap
    units = []
    for seg in segments:
        words = [w for w in (seg.get("words") or [])
                 if w.get("start") is not None and w.get("end") is not None]
        if not words:
            if seg.get("text", "").strip():
                units.append({"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()})
            continue
        cur = []
        for w in words:
            if cur and gap and (w["start"] - cur[-1]["end"] > gap):
                units.append(_mk_unit(cur))
                cur = []
            cur.append(w)
            if _SENT_END.search((w.get("word") or "").strip()):
                units.append(_mk_unit(cur))
                cur = []
        if cur:
            units.append(_mk_unit(cur))
    return [u for u in units if u["text"]]


def llm_relabel(segments) -> list:
    """Return turns [{speaker: 'agent'|'customer', start, end, text}] via LLM labeling.

    Raises on total failure so the caller can fall back to acoustic diarization.
    """
    units = sentence_units(segments)
    if not units:
        return []

    labels: dict[int, str | None] = {}
    for start in range(0, len(units), _CHUNK):
        chunk = units[start:start + _CHUNK]
        lines = "\n".join(f"{start + k}: {u['text']}" for k, u in enumerate(chunk))
        out = llm.chat_json(_PROMPT.format(lines=lines), system=SYSTEM, options=_OPTS,
                            model=settings.diarize_model())
        got = out.get("labels", {}) if isinstance(out, dict) else {}
        for k in range(len(chunk)):
            i = start + k
            raw = str(got.get(str(i), got.get(i, ""))).strip().lower()
            labels[i] = "customer" if raw.startswith("c") else ("agent" if raw.startswith("a") else None)

    if not any(v for v in labels.values()):
        raise ValueError("LLM returned no usable speaker labels")

    # Fill any line the model skipped by carrying the previous speaker forward.
    last = "agent"
    for i in range(len(units)):
        if labels.get(i) is None:
            labels[i] = last
        last = labels[i]

    turns = []
    for i, u in enumerate(units):
        spk = labels[i]
        if turns and turns[-1]["speaker"] == spk:
            turns[-1]["end"] = u["end"]
            turns[-1]["text"] += " " + u["text"]
        else:
            turns.append({"speaker": spk, "start": u["start"], "end": u["end"], "text": u["text"]})
    return turns
