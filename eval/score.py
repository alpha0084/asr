"""Score a transcription config against the hand-corrected references.

Usage:
    source env.sh && python eval/score.py --label baseline [--model large-v3] [--language auto]

Transcribes every dataset entry that has a corrected ref in eval/refs/, computes
WER/CER, and prints per-call + per-language + overall numbers. Saves the run to
eval/out/<label>/ (hypotheses + results.json) so you can compare configs.
"""
import argparse
import json
from collections import defaultdict

from lib import OUT, REFS, load_dataset, score, transcribe_text


def _avg(pairs):
    if not pairs:
        return 0.0, 0.0
    n = len(pairs)
    return sum(w for w, _ in pairs) / n, sum(c for _, c in pairs) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run", help="name for this config's output folder")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="auto",
                    help="force a source language, or 'auto' to detect / use per-entry")
    ap.add_argument("--clean", action="store_true", help="apply audio cleanup before ASR")
    args = ap.parse_args()
    lang_override = None if args.language.strip().lower() in ("auto", "") else args.language.strip()

    ds = load_dataset()
    outdir = OUT / args.label
    outdir.mkdir(parents=True, exist_ok=True)

    rows, by_lang = [], defaultdict(list)
    print(f"\nconfig: model={args.model} language={args.language}\n" + "-" * 60)
    for e in ds:
        ref_path = REFS / f"{e['id']}.txt"
        if not ref_path.exists():
            print(f"  {e['id']:22} — no corrected ref, skipping")
            continue
        hyp, det = transcribe_text(e, args.model, language=lang_override, clean=args.clean)
        (outdir / f"{e['id']}.txt").write_text(hyp + "\n")
        w, c = score(ref_path.read_text(), hyp)
        lang = e.get("language") or det or "?"
        rows.append({"id": e["id"], "language": lang, "wer": round(w, 4), "cer": round(c, 4)})
        by_lang[lang].append((w, c))
        print(f"  {e['id']:22} {lang:6} WER={w:6.1%}  CER={c:6.1%}")

    if not rows:
        print("\nNo scored calls. Run make_refs.py and correct the refs first.")
        return

    print("-" * 60)
    for lang, pairs in sorted(by_lang.items()):
        w, c = _avg(pairs)
        print(f"  {lang:6} ({len(pairs):2d} calls)   WER={w:6.1%}  CER={c:6.1%}")
    ow, oc = _avg([(r["wer"], r["cer"]) for r in rows])
    print("-" * 60)
    print(f"  OVERALL ({len(rows)} calls)   WER={ow:6.1%}  CER={oc:6.1%}\n")

    summary = {"label": args.label, "model": args.model, "language": args.language,
               "overall": {"wer": round(ow, 4), "cer": round(oc, 4), "calls": len(rows)},
               "per_call": rows}
    (outdir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved → {outdir/'results.json'}")


if __name__ == "__main__":
    main()
