"""Baseline eval: Malayalam audio -> English, scored vs the hand-corrected English
references (finetune/data/dataset.json) with BLEU + chrF (sacrebleu).

Systems (pluggable, select with --systems):
  whisper-translate          faster-whisper large-v3, task=translate (direct speech->EN)
  indicconformer-indictrans2 IndicConformer (ml, CTC, 30s chunks) -> IndicTrans2 (ml->en)
  seamless                   SeamlessM4T-medium speech->text translation (30s chunks)

RESUMABLE: every (system, call) hypothesis is written to finetune/out/<system>/<id>.txt
and reused if present — so a killed run just continues. Runs off-box; long inputs are
cut to 30s chunks (both new models + to avoid seq2seq repetition on long blobs).
"""
import argparse, json, os, subprocess, re
from pathlib import Path

# hf_xet is broken on the box ("process() takes no keyword arguments" on every Xet repo);
# force classic HTTPS downloads. Must be set before any huggingface_hub/transformers import.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent
DATA, OUT, WAV, CHUNKS = ROOT/"data", ROOT/"out", ROOT/"data"/"wav", ROOT/"data"/"chunks"
for d in (OUT, WAV, CHUNKS): d.mkdir(parents=True, exist_ok=True)
import sacrebleu
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"   # use the box GPU when present, else CPU


def to_wav(mpeg, wav):
    if not Path(wav).exists():
        subprocess.run(["ffmpeg","-y","-i",mpeg,"-ar","16000","-ac","1",str(wav)], capture_output=True)

def chunks_30s(wav, cid):
    d = CHUNKS/cid; d.mkdir(exist_ok=True)
    if not any(d.glob("*.wav")):
        subprocess.run(["ffmpeg","-y","-i",str(wav),"-f","segment","-segment_time","30",
                        "-ar","16000","-ac","1",str(d/"c_%03d.wav")], capture_output=True)
    return sorted(d.glob("*.wav"))

# ---------------- systems ----------------
_wm=None
def whisper_translate(wav, cid):
    """transformers Whisper large-v3, task=translate -> English directly. Uses the torch
    stack (GPU-aware), so no faster-whisper/CTranslate2 CUDA-lib juggling."""
    global _wm
    if _wm is None:
        from transformers import pipeline
        _wm = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                       chunk_length_s=30, device=(0 if DEV == "cuda" else -1),
                       torch_dtype=(torch.float16 if DEV == "cuda" else torch.float32))
    r = _wm(str(wav), generate_kwargs={"task": "translate"})
    return (r.get("text") or "").strip()

_ic=None
def _asr_ml(chunk):
    global _ic
    if _ic is None:
        from transformers import AutoModel
        _ic = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual",
                                        trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    import torch, soundfile as sf
    d,sr = sf.read(str(chunk), dtype="float32")
    if d.ndim>1: d=d.mean(1)
    return _ic(torch.from_numpy(d).float().unsqueeze(0), "ml", "ctc")

_it=None
def _mt_ml_en(text):
    """Malayalam text -> English via NLLB-200, translated in ~25-word windows so each
    input is sentence-sized (IndicConformer output is unpunctuated); no_repeat_ngram_size
    guards against the long-blob repetition degeneration we saw earlier."""
    global _it
    if _it is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(DEV)
        _it = (tok, mdl, tok.convert_tokens_to_ids("eng_Latn"), torch)
    tok, mdl, EN, torch = _it
    tok.src_lang = "mal_Mlym"
    w = text.split(); units = [" ".join(w[i:i+25]) for i in range(0, len(w), 25)] or [text]
    outs = []
    for u in units:
        inp = tok(u, return_tensors="pt", truncation=True, max_length=128).to(DEV)
        with torch.no_grad():
            o = mdl.generate(**inp, forced_bos_token_id=EN, max_length=160,
                             num_beams=4, no_repeat_ngram_size=3)
        outs.append(tok.batch_decode(o, skip_special_tokens=True)[0])
    return " ".join(outs)

def indicconformer_indictrans2(wav, cid):
    en=[]
    for ch in chunks_30s(wav, cid):
        ml = _asr_ml(ch)
        if ml and ml.strip(): en.append(_mt_ml_en(ml))
    return " ".join(en).strip()

_sm=None
def seamless(wav, cid):
    global _sm
    if _sm is None:
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        mdl = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large").to(DEV)
        _sm = (proc, mdl)
    import soundfile as sf
    proc, mdl = _sm; en=[]
    for ch in chunks_30s(wav, cid):
        d,sr = sf.read(str(ch), dtype="float32")
        if d.ndim>1: d=d.mean(1)
        inp = proc(audio=d, sampling_rate=16000, return_tensors="pt").to(DEV)
        out = mdl.generate(**inp, tgt_lang="eng")[0].cpu().tolist()
        en.append(proc.decode(out, skip_special_tokens=True))
    return " ".join(en).strip()

SYSTEMS = {"whisper-translate": whisper_translate,
           "indicconformer-nllb": indicconformer_indictrans2,
           "seamless": seamless}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--ids", default="", help="comma-sep call ids to run (subset); blank = all")
    args = ap.parse_args()
    want = [s.strip() for s in args.systems.split(",") if s.strip() in SYSTEMS]
    only = {i.strip() for i in args.ids.split(",") if i.strip()}
    ds = [e for e in json.loads((DATA/"dataset.json").read_text())
          if e.get("audio") and (not only or e["id"] in only)]
    for name in want:
        fn = SYSTEMS[name]; outdir = OUT/name; outdir.mkdir(parents=True, exist_ok=True)
        hyps, refs, rows = [], [], []
        print(f"\n=== system: {name} ===", flush=True)
        for e in ds:
            wav = WAV/f"{e['id']}.wav"; to_wav(e["audio"], wav)
            hp = outdir/f"{e['id']}.txt"
            try:
                hyp = hp.read_text() if hp.exists() else fn(wav, e["id"])
                if not hp.exists(): hp.write_text(hyp)
            except Exception as ex:
                print(f"  {e['id']:12} ERROR: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
                continue
            ref = e["ref_en"]
            b = sacrebleu.sentence_bleu(hyp, [ref]).score
            c = sacrebleu.sentence_chrf(hyp, [ref]).score
            rows.append({"id": e["id"], "bleu": round(b,1), "chrf": round(c,1),
                         "hyp_chars": len(hyp), "ref_chars": len(ref)})
            hyps.append(hyp); refs.append(ref)
            print(f"  {e['id']:12} BLEU={b:5.1f}  chrF={c:5.1f}  hyp_chars={len(hyp):5d}", flush=True)
        if not hyps:
            print(f"[{name}] no results", flush=True); continue
        cB = sacrebleu.corpus_bleu(hyps,[refs]).score
        cC = sacrebleu.corpus_chrf(hyps,[refs]).score
        print(f"[{name}] CORPUS  BLEU={cB:.1f}  chrF={cC:.1f}  ({len(hyps)} calls)", flush=True)
        (outdir/"results.json").write_text(json.dumps(
            {"system": name, "corpus": {"bleu": round(cB,1), "chrf": round(cC,1)}, "per_call": rows}, indent=2))


if __name__ == "__main__":
    main()
