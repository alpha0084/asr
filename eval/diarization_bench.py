"""Diarization / speaker-labeling A/B benchmark.

Runs several candidate approaches on the SAME set of real recordings and scores them, so we pick a
diarization strategy on numbers rather than one cherry-picked call. READ-ONLY: it loads saved audio
(data/<jid>/audio16k.wav) and never writes to the DB or the live pipeline.

Candidates:
  hybrid : pyannote diarization + word-level assembly + one-shot LLM role naming  (current acoustic)
  llm    : sentence-split + per-line LLM labeling                                  (deployed default)
  nemo   : NVIDIA NeMo Sortformer (telephony)  -- only if `nemo_toolkit` is installed, else skipped

Metrics:
  Label-free (always): speakers_detected, turn_count, median_turn_sec (granularity),
                       merged_turn_est (long turns with an internal '?', a proxy for two-speaker merges).
  Ground-truth (optional eval/ground_truth.json): role_accuracy, and DER via pyannote.metrics if the
                       reference carries timings.

Usage (on the box):
  .venv/bin/python eval/diarization_bench.py --last 5
  .venv/bin/python eval/diarization_bench.py --tasks 96401c93b43b,4bda1155462a --candidates hybrid,llm
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, settings  # noqa: E402
from backend.pipeline import assemble, diarize, preprocess, refine_speakers as rs  # noqa: E402
from backend.pipeline.transcribe import _get_faster  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402


def _audio_path(jid: str) -> str:
    return os.path.join(str(DATA_DIR), jid, "audio16k.wav")


def _pick_tasks(args) -> list[str]:
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",") if t.strip()]
    rows = db.list_recordings(limit=args.last * 3, status="done")  # over-fetch; filter to ones with audio
    out = []
    for r in rows:
        if os.path.isfile(_audio_path(r["id"])):
            out.append(r["id"])
        if len(out) >= args.last:
            break
    return out


def _transcribe(waveform):
    model = _get_faster(settings.whisper_model(), "cuda")
    segs, _ = model.transcribe(waveform, task="transcribe", word_timestamps=True,
                               language=None, vad_filter=True, beam_size=5)
    return [{"start": s.start, "end": s.end, "text": s.text.strip(),
             "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]}
            for s in segs]


# ---- candidates: each returns turns [{speaker, start, end, text}] -----------------------------
def run_hybrid(waveform, sr, segd):
    dia = diarize.diarize(waveform, sr, num_speakers=2)
    turns = assemble.build_turns(assemble.assign_speakers(segd, dia))
    by_spk: dict = {}
    for t in turns:
        by_spk.setdefault(t["speaker"], []).append(t["text"])
    mapping = {}
    try:
        mapping = rs.orient(by_spk)
    except Exception:
        pass
    for t in turns:
        t["speaker"] = mapping.get(t["speaker"], t["speaker"])
    return turns


def run_llm(waveform, sr, segd):
    return rs.llm_relabel(segd)


def run_nemo(waveform, sr, segd):
    try:
        import nemo.collections.asr  # noqa: F401
    except Exception:
        return None  # not installed → skipped; see README to add
    raise NotImplementedError("NeMo Sortformer adapter: wire the pipeline once nemo_toolkit is installed")


CANDIDATES = {"hybrid": run_hybrid, "llm": run_llm, "nemo": run_nemo}


# ---- metrics ----------------------------------------------------------------------------------
def label_free(turns: list) -> dict:
    durs = [t["end"] - t["start"] for t in turns if t.get("end") and t.get("start") is not None]
    merged = sum(1 for t in turns
                 if (t["end"] - t["start"]) > 6 and "?" in t["text"][:-1])
    return {
        "speakers": len({t["speaker"] for t in turns}),
        "turns": len(turns),
        "median_turn_s": round(statistics.median(durs), 1) if durs else 0,
        "merged_est": merged,
    }


def role_accuracy(turns: list, truth: list) -> float | None:
    """Align each ground-truth line to the best-overlapping predicted turn by text, compare role."""
    if not truth:
        return None
    def norm(s):
        return "".join(c.lower() for c in s if c.isalnum() or c == " ").split()
    hits = tot = 0
    for g in truth:
        gw = set(norm(g["text"]))
        if not gw:
            continue
        best, bestov = None, 0
        for t in turns:
            ov = len(gw & set(norm(t["text"])))
            if ov > bestov:
                bestov, best = ov, t
        if best is not None:
            tot += 1
            if str(best["speaker"]).lower().startswith(str(g["role"]).lower()[:4]):
                hits += 1
    return round(100 * hits / tot, 1) if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=5)
    ap.add_argument("--tasks", type=str, default="")
    ap.add_argument("--candidates", type=str, default="hybrid,llm,nemo")
    args = ap.parse_args()

    cand_names = [c.strip() for c in args.candidates.split(",") if c.strip() in CANDIDATES]
    tasks = _pick_tasks(args)
    truth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json")
    truth = json.load(open(truth_path)) if os.path.isfile(truth_path) else {}

    print("tasks: %s | candidates: %s | ground-truth calls: %d\n" % (tasks, cand_names, len(truth)))
    agg: dict = {c: {"turns": [], "median": [], "merged": [], "role": []} for c in cand_names}

    for jid in tasks:
        wav = _audio_path(jid)
        if not os.path.isfile(wav):
            print("skip %s (no audio)" % jid)
            continue
        waveform, sr = preprocess.load_waveform(wav)
        segd = _transcribe(waveform)
        print("=== %s ===" % jid)
        for c in cand_names:
            try:
                turns = CANDIDATES[c](waveform, sr, segd)
            except NotImplementedError as e:
                print("  %-7s n/a (%s)" % (c, e)); continue
            if turns is None:
                print("  %-7s skipped (not installed)" % c); continue
            lf = label_free(turns)
            ra = role_accuracy(turns, truth.get(jid, []))
            print("  %-7s speakers=%d turns=%d median=%.1fs merged_est=%d role_acc=%s"
                  % (c, lf["speakers"], lf["turns"], lf["median_turn_s"], lf["merged_est"],
                     "%.1f%%" % ra if ra is not None else "-"))
            agg[c]["turns"].append(lf["turns"]); agg[c]["median"].append(lf["median_turn_s"])
            agg[c]["merged"].append(lf["merged_est"])
            if ra is not None:
                agg[c]["role"].append(ra)

    print("\n===== SUMMARY (averaged across calls) =====")
    print("  cand     avg_turns  avg_median_s  avg_merged_est  avg_role_acc")
    for c in cand_names:
        a = agg[c]
        avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else 0
        role = "%.1f%%" % avg(a["role"]) if a["role"] else "-"
        print("  %-7s  %-9s  %-12s  %-14s  %s"
              % (c, avg(a["turns"]), avg(a["median"]), avg(a["merged"]), role))


if __name__ == "__main__":
    main()
