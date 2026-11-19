"""Baseline eval: Malayalam audio -> English, scored against the hand-corrected
English references (finetune/data/dataset.json) with BLEU + chrF (sacrebleu).

Systems are pluggable (SYSTEMS dict). This first pass runs `whisper-translate`
(faster-whisper large-v3, task='translate' = direct speech->English). SeamlessM4T
and IndicConformer->LLM can be added as more entries later.

Runs entirely off-box. Outputs per-system hypotheses + results.json under finetune/out/.
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
WAV = DATA / "wav"
WAV.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

import sacrebleu


def to_wav(mpeg: str, wav: Path):
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-i", mpeg, "-ar", "16000", "-ac", "1", str(wav)],
                       capture_output=True)


# ---------------- systems ----------------
_wm = None
def whisper_translate(wav: str) -> str:
    """faster-whisper large-v3, task=translate -> English directly."""
    global _wm
    from faster_whisper import WhisperModel
    if _wm is None:
        _wm = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segs, _ = _wm.transcribe(wav, task="translate", beam_size=5, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()


SYSTEMS = {"whisper-translate": whisper_translate}


def main():
    ds = json.loads((DATA / "dataset.json").read_text())
    ds = [e for e in ds if e.get("audio")]
    for name, fn in SYSTEMS.items():
        outdir = OUT / name
        outdir.mkdir(parents=True, exist_ok=True)
        hyps, refs, rows = [], [], []
        print(f"\n=== system: {name} ===", flush=True)
        for e in ds:
            wav = WAV / f"{e['id']}.wav"
            to_wav(e["audio"], wav)
            hyp = fn(str(wav))
            ref = e["ref_en"]
            (outdir / f"{e['id']}.txt").write_text(hyp)
            b = sacrebleu.sentence_bleu(hyp, [ref]).score
            c = sacrebleu.sentence_chrf(hyp, [ref]).score
            rows.append({"id": e["id"], "bleu": round(b, 1), "chrf": round(c, 1),
                         "hyp_chars": len(hyp), "ref_chars": len(ref)})
            hyps.append(hyp); refs.append(ref)
            print(f"  {e['id']:12} BLEU={b:5.1f}  chrF={c:5.1f}  hyp_chars={len(hyp):5d}", flush=True)
        cB = sacrebleu.corpus_bleu(hyps, [refs]).score
        cC = sacrebleu.corpus_chrf(hyps, [refs]).score
        print(f"\n[{name}] CORPUS  BLEU={cB:.1f}  chrF={cC:.1f}  ({len(hyps)} calls)", flush=True)
        (outdir / "results.json").write_text(json.dumps(
            {"system": name, "corpus": {"bleu": round(cB, 1), "chrf": round(cC, 1)},
             "per_call": rows}, indent=2))


if __name__ == "__main__":
    main()
