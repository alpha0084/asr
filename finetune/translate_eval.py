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

ROOT = Path(__file__).resolve().parent
DATA, OUT, WAV, CHUNKS = ROOT/"data", ROOT/"out", ROOT/"data"/"wav", ROOT/"data"/"chunks"
for d in (OUT, WAV, CHUNKS): d.mkdir(parents=True, exist_ok=True)
import sacrebleu


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
    global _wm
    from faster_whisper import WhisperModel
    if _wm is None: _wm = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segs,_ = _wm.transcribe(str(wav), task="translate", beam_size=5, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()

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
    global _it
    if _it is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor
        tok = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)
        _it = (tok, mdl, IndicProcessor(inference=True), torch)
    tok, mdl, ip, torch = _it
    # ~25-word windows so each MT input is sentence-sized (IndicConformer output is unpunctuated)
    w = text.split(); units = [" ".join(w[i:i+25]) for i in range(0, len(w), 25)] or [text]
    batch = ip.preprocess_batch(units, src_lang="mal_Mlym", tgt_lang="eng_Latn")
    inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        out = mdl.generate(**inp, max_length=256, num_beams=5)
    dec = tok.batch_decode(out, skip_special_tokens=True)
    return " ".join(ip.postprocess_batch(dec, lang="eng_Latn"))

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
        from transformers import AutoProcessor, SeamlessM4TForSpeechToText
        proc = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
        mdl = SeamlessM4TForSpeechToText.from_pretrained("facebook/hf-seamless-m4t-medium")
        _sm = (proc, mdl)
    import soundfile as sf
    proc, mdl = _sm; en=[]
    for ch in chunks_30s(wav, cid):
        d,sr = sf.read(str(ch), dtype="float32")
        if d.ndim>1: d=d.mean(1)
        inp = proc(audios=d, sampling_rate=16000, return_tensors="pt")
        out = mdl.generate(**inp, tgt_lang="eng")[0].tolist()
        en.append(proc.decode(out, skip_special_tokens=True))
    return " ".join(en).strip()

SYSTEMS = {"whisper-translate": whisper_translate,
           "indicconformer-indictrans2": indicconformer_indictrans2,
           "seamless": seamless}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    args = ap.parse_args()
    want = [s.strip() for s in args.systems.split(",") if s.strip() in SYSTEMS]
    ds = [e for e in json.loads((DATA/"dataset.json").read_text()) if e.get("audio")]
    for name in want:
        fn = SYSTEMS[name]; outdir = OUT/name; outdir.mkdir(parents=True, exist_ok=True)
        hyps, refs, rows = [], [], []
        print(f"\n=== system: {name} ===", flush=True)
        for e in ds:
            wav = WAV/f"{e['id']}.wav"; to_wav(e["audio"], wav)
            hp = outdir/f"{e['id']}.txt"
            hyp = hp.read_text() if hp.exists() else fn(wav, e["id"])
            if not hp.exists(): hp.write_text(hyp)
            ref = e["ref_en"]
            b = sacrebleu.sentence_bleu(hyp, [ref]).score
            c = sacrebleu.sentence_chrf(hyp, [ref]).score
            rows.append({"id": e["id"], "bleu": round(b,1), "chrf": round(c,1),
                         "hyp_chars": len(hyp), "ref_chars": len(ref)})
            hyps.append(hyp); refs.append(ref)
            print(f"  {e['id']:12} BLEU={b:5.1f}  chrF={c:5.1f}  hyp_chars={len(hyp):5d}", flush=True)
        cB = sacrebleu.corpus_bleu(hyps,[refs]).score
        cC = sacrebleu.corpus_chrf(hyps,[refs]).score
        print(f"[{name}] CORPUS  BLEU={cB:.1f}  chrF={cC:.1f}  ({len(hyps)} calls)", flush=True)
        (outdir/"results.json").write_text(json.dumps(
            {"system": name, "corpus": {"bleu": round(cB,1), "chrf": round(cC,1)}, "per_call": rows}, indent=2))


if __name__ == "__main__":
    main()
