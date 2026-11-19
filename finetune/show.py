"""Eyeball eval output: for each call id, print the English reference next to each
system's hypothesis. A low BLEU can mean the model is wrong OR that the human English
reference is simply worded differently from a literal transcription — only reading the
text tells them apart.

    python finetune/show.py sreethu_1 nexora_2      # specific calls
    python finetune/show.py                         # all calls
    python finetune/show.py --chars 2500 nexora_2   # longer excerpt
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="call ids to show; blank = all")
    ap.add_argument("--chars", type=int, default=1400, help="excerpt length per block")
    args = ap.parse_args()

    ds = {e["id"]: e for e in json.loads((DATA / "dataset.json").read_text())}
    systems = sorted(p.name for p in OUT.iterdir() if p.is_dir()) if OUT.exists() else []
    ids = args.ids or list(ds)
    n = args.chars

    for cid in ids:
        e = ds.get(cid)
        if not e:
            print(f"\n## {cid}: not in dataset"); continue
        print(f"\n{'='*100}\n## {cid}   audio {e.get('duration_s','?')}s   ref {len(e['ref_en'])} chars\n{'='*100}")
        print(f"\n----- REFERENCE (human English) -----\n{e['ref_en'][:n]}")
        for s in systems:
            hp = OUT / s / f"{cid}.txt"
            full = hp.read_text() if hp.exists() else ""
            print(f"\n----- {s}  ({len(full)} chars) -----\n{full[:n] or '(no output)'}")


if __name__ == "__main__":
    main()
