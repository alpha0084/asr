"""Generate DRAFT reference transcripts for the dataset, for you to hand-correct.

Usage:
    source env.sh && python eval/make_refs.py [--model large-v3]

For each entry in eval/dataset.json it writes eval/refs/<id>.txt (only if missing,
so it never overwrites your corrections). Then you EDIT each file to fix errors —
those corrected files become the ground truth used by score.py.
"""
import argparse

from lib import REFS, load_dataset, transcribe_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3", help="whisper model for the draft")
    args = ap.parse_args()

    ds = load_dataset()
    if not ds:
        print("eval/dataset.json is empty — add entries first (see eval/README.md).")
        return

    REFS.mkdir(parents=True, exist_ok=True)
    for e in ds:
        ref = REFS / f"{e['id']}.txt"
        if ref.exists():
            print(f"skip (already have ref): {e['id']}")
            continue
        print(f"transcribing draft: {e['id']} …", flush=True)
        text, lang = transcribe_text(e, args.model)
        ref.write_text(text + "\n")
        print(f"  wrote {ref}  (lang={lang}) — NOW EDIT IT to correct errors")


if __name__ == "__main__":
    main()
