# Diarization / speaker-labeling benchmark

Compares candidate speaker-separation strategies on the SAME real recordings so the choice (keep
pyannote-hybrid, adopt NeMo, keep the LLM mode, or a hybrid) is made on numbers. Read-only — runs on
saved `data/<jid>/audio16k.wav`, never touches the DB or the live pipeline.

## Run (on the GPU box, as the `asr` user)

```
cd /state/asr
export HOME=/state/asr-home HF_HOME=/state/asr/data/hf XDG_CACHE_HOME=/state/asr/data/cache
.venv/bin/python eval/diarization_bench.py --last 5
# or specific calls / candidates:
.venv/bin/python eval/diarization_bench.py --tasks 96401c93b43b,4bda1155462a --candidates hybrid,llm
```

## Candidates
- `hybrid` — pyannote diarization + word-level assembly + one-shot LLM role naming (current acoustic).
- `llm`    — sentence-split + per-line LLM labeling (the deployed default).
- `nemo`   — NVIDIA NeMo Sortformer (telephony). Skipped automatically unless `nemo_toolkit` is
             installed; the adapter in `diarization_bench.py` must be wired once it is (see below).

## Metrics
- Always (label-free): `speakers` (should be 2), `turns`, `median_turn_s` (finer = more turns of
  shorter duration), `merged_est` (long turns containing an internal `?` — a proxy for a two-speaker
  exchange collapsed onto one speaker).
- With ground truth (`eval/ground_truth.json`): `role_accuracy`, and DER via `pyannote.metrics` if
  the reference includes timings.

## Ground truth
Create `eval/ground_truth.json` mapping task id → ordered lines with the correct role:

```json
{
  "96401c93b43b": [
    {"role": "customer", "text": "Hello."},
    {"role": "agent", "text": "Hello, am I speaking with Dasutosh?"},
    {"role": "customer", "text": "Yes, you are speaking with Dasutosh."}
  ]
}
```

Only the calls you label get a `role_accuracy`; the rest still report the label-free diagnostics.

## Adding NeMo (heavy — do NOT install into the production venv)
NeMo pulls a large dependency tree and extra VRAM. Install it into a SEPARATE venv/container, wire
`run_nemo()` to produce `[{speaker, start, end, text}]` from Sortformer, then run the benchmark from
that venv. Measure peak VRAM alongside whisper + the Ollama models before considering it for prod.
