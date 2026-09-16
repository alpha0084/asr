"""Stage 4 — merge transcription + diarization into speaker-labeled turns + talk stats."""
import difflib
import re
from collections import defaultdict


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _norm_text(s: str) -> str:
    """Lowercase, keep word chars + Devanagari, collapse whitespace — for comparing
    two transcript segments regardless of punctuation/spacing."""
    return re.sub(r"\s+", " ", re.sub(r"[^\wऀ-ॿ]+", " ", (s or "").lower())).strip()


def _seg_text(seg: dict) -> str:
    txt = seg.get("text")
    if not txt:
        txt = " ".join((w.get("word") or "") for w in (seg.get("words") or []))
    return txt


def dedupe_overlapping_segments(segments: list) -> list:
    """Drop whisper segments that are near-duplicates of a recent segment AND overlap
    it in time.

    On low-quality (8kHz mono) audio Whisper sometimes emits the same utterance twice
    over an overlapping time span; diarization then assigns the two copies to different
    speakers, surfacing as one utterance repeated under both Agent and Customer. We drop
    the later copy. The TIME-OVERLAP requirement is what distinguishes this artifact from
    a legitimate sequential repeat (e.g. one speaker echoing another) — genuine repeats
    are consecutive in time, not overlapping — so those are kept.
    """
    kept: list = []
    for seg in segments:
        s0, s1 = seg.get("start"), seg.get("end")
        txt = _norm_text(_seg_text(seg))
        is_dup = False
        if txt and len(txt) >= 8 and s0 is not None and s1 is not None and s1 > s0:
            for k in kept[-3:]:                       # only compare against near neighbours
                k0, k1 = k.get("start"), k.get("end")
                if k0 is None or k1 is None or k1 <= k0:
                    continue
                ov = _overlap(s0, s1, k0, k1)
                shorter = min(s1 - s0, k1 - k0) or 1e-9
                if ov / shorter < 0.5:                # must overlap in time to be a duplicate
                    continue
                ktxt = _norm_text(_seg_text(k))
                if (txt in ktxt or ktxt in txt
                        or difflib.SequenceMatcher(None, txt, ktxt).ratio() >= 0.8):
                    is_dup = True
                    break
        if not is_dup:
            kept.append(seg)
    return kept


def _speaker_for(t0: float, t1: float, dia_turns: list) -> str | None:
    """Diarization speaker whose region overlaps [t0, t1] the most.

    When a word falls in a gap pyannote left unlabelled (no overlap with any
    region), attribute it to the nearest region in time rather than inventing a
    phantom "unknown" speaker.
    """
    best_spk, best_ov = None, 0.0
    for t in dia_turns:
        ov = _overlap(t0, t1, t["start"], t["end"])
        if ov > best_ov:
            best_ov, best_spk = ov, t["speaker"]
    if best_spk is not None or not dia_turns:
        return best_spk
    mid = (t0 + t1) / 2.0

    def _gap(t):
        if mid < t["start"]:
            return t["start"] - mid
        if mid > t["end"]:
            return mid - t["end"]
        return 0.0

    return min(dia_turns, key=_gap)["speaker"]


def _join_words(tokens: list) -> str:
    """Reconstruct text from word tokens, backend-agnostic about leading spaces.

    faster-whisper emits words with a leading space; whisperx/mlx may not. We
    strip each token and re-space, keeping trailing punctuation attached to the
    preceding word.
    """
    out = ""
    for raw in tokens:
        tok = (raw or "").strip()
        if not tok:
            continue
        if out and tok[0] not in ",.?!।:;%)]}’”\"'":
            out += " "
        out += tok
    return out


def _smooth_runs(runs: list) -> list:
    """Reassign lone single-word islands to a flanking speaker.

    Word-level diarization can flip one short word to the other speaker on a
    boundary; if a tiny run sits between two runs of the SAME other speaker,
    fold it into them so we don't emit a spurious one-word turn.
    """
    for i in range(1, len(runs) - 1):
        r = runs[i]
        short = (r["end"] - r["start"] <= 0.6) and len(r["tokens"]) <= 1
        if (short and runs[i - 1]["speaker"] == runs[i + 1]["speaker"]
                and runs[i - 1]["speaker"] != r["speaker"]):
            r["speaker"] = runs[i - 1]["speaker"]
    return runs


def assign_speakers(segments: list, dia_turns: list) -> list:
    """Tag speech with speakers at the WORD level.

    Whisper splits segments on pauses/punctuation, not on speaker changes, so a
    single segment can straddle a speaker switch. Assigning one speaker per
    segment (the old behaviour) merges both people into one turn. Instead we
    label each word by the diarization region it overlaps and split the segment
    wherever the speaker changes between words. Segments without usable word
    timestamps fall back to segment-level max-overlap.
    """
    runs: list = []

    def flush(run):
        if run and run["tokens"]:
            text = _join_words(run["tokens"])
            if text:
                runs.append(run)

    for seg in dedupe_overlapping_segments(segments):
        words = [
            w for w in (seg.get("words") or [])
            if w.get("start") is not None and w.get("end") is not None
            and (w.get("word") or "").strip()
        ]
        if not words:
            # No word timestamps — keep the whole segment as one speaker.
            spk = _speaker_for(seg["start"], seg["end"], dia_turns) or "SPEAKER_?"
            text = (seg.get("text") or "").strip()
            if text:
                runs.append({"start": seg["start"], "end": seg["end"],
                             "speaker": spk, "tokens": [text]})
            continue

        run = None
        prev_spk = None
        for w in words:
            spk = (_speaker_for(w["start"], w["end"], dia_turns)
                   or prev_spk
                   or _speaker_for(seg["start"], seg["end"], dia_turns)
                   or "SPEAKER_?")
            prev_spk = spk
            if run and run["speaker"] == spk:
                run["end"] = w["end"]
                run["tokens"].append(w["word"])
            else:
                flush(run)
                run = {"start": w["start"], "end": w["end"],
                       "speaker": spk, "tokens": [w["word"]]}
        flush(run)

    _smooth_runs(runs)

    return [
        {"start": r["start"], "end": r["end"], "speaker": r["speaker"],
         "text": _join_words(r["tokens"])}
        for r in runs
    ]


def build_turns(segments: list) -> list:
    """Collapse consecutive same-speaker segments into readable turns."""
    turns = []
    for seg in segments:
        if not seg["text"]:
            continue
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turns[-1]["end"] = seg["end"]
            turns[-1]["text"] += " " + seg["text"]
        else:
            turns.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })
    return turns


def talk_stats(dia_turns: list, duration: float) -> dict:
    """Per-speaker talk time, share, and talk-to-listen ratio (from diarization)."""
    talk = defaultdict(float)
    for t in dia_turns:
        talk[t["speaker"]] += t["end"] - t["start"]
    total_talk = sum(talk.values()) or 1.0
    per_speaker = {
        spk: {
            "talk_seconds": round(secs, 1),
            "talk_share_pct": round(100 * secs / total_talk, 1),
        }
        for spk, secs in sorted(talk.items())
    }
    return {
        "duration_seconds": round(duration, 1),
        "speech_seconds": round(total_talk, 1),
        "silence_seconds": round(max(0.0, duration - total_talk), 1),
        "num_speakers": len(talk),
        "per_speaker": per_speaker,
    }
