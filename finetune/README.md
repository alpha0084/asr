# Malayalam‑audio → English: eval + fine‑tune (RTX 5090)

Turnkey environment for choosing and fine‑tuning a speech‑translation model
(Malayalam audio → English). Isolated from the prod ASR stack.

## 0. Environment (one‑time, reproducible)

The RTX 5090 (Blackwell / sm_120) **requires torch 2.7 + CUDA 12.8**. Build the image once:

```powershell
cd D:\project\asr-finetune
docker build -f finetune\Dockerfile -t asr-finetune finetune
```

Run a shell in it (mounts code, data, and a persistent HF model cache):

```powershell
docker run --rm -it --gpus all `
  -v D:\project\asr-finetune:/work -w /work `
  -v D:\project\ft-data:/data `
  -v D:\project\ft-hfcache:/root/.cache/huggingface `
  -e HF_TOKEN=<hf_token> -e ASR_FT_SRC=/data `
  asr-finetune bash
# verify the GPU is usable:
python -c "import torch;x=torch.randn(8,device='cuda');print('5090 OK',(x*2).sum().item() is not None)"
```

## 1. Data → manifest

Put your labeled data under `D:\project\ft-data\` (= `/data`), then:

```bash
python finetune/build_dataset.py        # reads $ASR_FT_SRC → finetune/data/dataset.json
```
Manifest entries: `{id, audio, ref_en, turns, duration_s}`.

## 2. Eval — pick the best base model

```bash
python finetune/translate_eval.py --systems whisper-translate,indicconformer-nllb,seamless
```
Prints BLEU/chrF per system vs your English references; outputs under `finetune/out/`.

## 3. Fine‑tune — "pass the path and go"

```bash
python finetune/train.py --data finetune/data/dataset.json --out finetune/model \
    --base openai/whisper-large-v3 --src-lang ml --epochs 5 --lora
```
Then re‑run step 2 pointing at `finetune/model` to measure the gain.

---

## ⚠️ Data‑shape requirement (read before training)

Whisper sees **≤ 30 s per example**. A training example must be a **short clip (≤30 s)
with the English for THAT clip**. A full call (minutes long) with one call‑level English
transcript is an **eval** example, not a **train** example — it has to be segmented into
≤30 s clips with aligned English first.

So when you collect training data (e.g. via the QC feedback‑correction loop), capture it
as **short clips (per‑turn or ≤30 s) + their English**, and training is a one‑command run.
If you only have call‑level transcripts, we add a segmentation+alignment prep step (forced
alignment) before `train.py`.

**Volume:** meaningful fine‑tuning needs **tens of hours** (hundreds–thousands of clips).
The current 7 calls are a test set — enough to *choose* a base model, not to *train* one.
