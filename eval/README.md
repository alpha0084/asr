# Transcription accuracy eval harness

Measures **WER** (word error rate) and **CER** (character error rate) so we can tell
whether an accuracy change actually helps — instead of guessing. ASR only (no
diarization/LLM), so it's fast.

## Workflow

### 1. Define the test set — `eval/dataset.json`
A list of calls: `{ "id", "audio" (URL or local path), "language" (or "auto"), "note" }`.
Pick **6–10 representative calls** across the languages you actually get. (Pre-filled
with the calls tested so far — edit/extend it.)

### 2. Generate draft transcripts to correct
```bash
source env.sh
python eval/make_refs.py --model large-v3
```
Writes `eval/refs/<id>.txt` for each call (skips any that already exist).

### 3. Hand-correct each `eval/refs/<id>.txt`  ← the important human step
Open each file and fix every mistake while listening to the audio, until it's a
**perfect transcript**. These become the ground truth. (This is the one step only
you can do — accuracy can't be measured without it.)

### 4. Score a config
```bash
source env.sh
python eval/score.py --label baseline --model large-v3
```
Prints per-call, per-language, and overall WER/CER, and saves
`eval/out/baseline/` (hypotheses + `results.json`).

### 5. Compare changes
After each enhancement (denoise, prompting, etc.), re-run with a new label and
compare overall/per-language WER:
```bash
python eval/score.py --label denoise --model large-v3
python eval/score.py --label small   --model small
```
Lower WER/CER = better. Only keep changes that improve the numbers.

## Notes
- Scores are computed after normalization (lowercase, no punctuation, collapsed spaces).
- Refs are never overwritten by `make_refs.py`, so your corrections are safe.
- `eval/refs/` and `eval/out/` are yours to keep; audio is cached under `data/_eval/`.
