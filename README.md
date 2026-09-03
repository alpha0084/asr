# ASR Call Analytics Tool

Local, zero-cost tool to transcribe call recordings (any language → any
language), separate speakers, summarize, and score call quality with a standard
QA scorecard. Runs entirely on your Mac — nothing leaves the machine.

See [`TECHNICAL_APPROACH.md`](./TECHNICAL_APPROACH.md) for the full design.

## Setup

```bash
# 1. System tools (one time)
brew install ffmpeg ollama python@3.11

# 2. Python environment (Python 3.11 — required; 3.14 is too new for the ML stack)
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Local LLM
ollama serve            # in one terminal (or runs as a service)
ollama pull qwen2.5:7b

# 4. HuggingFace token for speaker diarization (free, one time)
#    - Make a free account at https://huggingface.co
#    - Create a read token: https://huggingface.co/settings/tokens
#    - Accept the model license: https://huggingface.co/pyannote/speaker-diarization-community-1
cp .env.example .env    # then paste your HF_TOKEN into .env
```

## Activate the environment

Before running anything, source the env helper (activates the venv):

```bash
source env.sh
```

> Audio decoding note: Homebrew ships ffmpeg 9, which `torchcodec` (pyannote's
> default audio loader) can't link (it only supports ffmpeg 4–7). Rather than
> juggle a second ffmpeg, the pipeline preprocesses audio to 16kHz mono WAV with
> the ffmpeg 9 CLI and loads it in-memory via `soundfile`, handing waveforms
> directly to Whisper and pyannote. No torchcodec, no library conflicts.

## Run (Phase 1 — speaker-wise transcript)

```bash
source env.sh

# local file (any audio/video format)
python -m backend.cli data/call.mp3 --speakers 2

# quick first test with a small/fast model, then use large-v3 for quality
python -m backend.cli data/call.m4a --model small --speakers 2

# from a presigned / public S3 (or any https) URL
python -m backend.cli "https://bucket.s3.amazonaws.com/call.wav?X-Amz-..." --speakers 2

# pin the source language (else auto-detected), e.g. Hindi
python -m backend.cli data/call.mp3 --language hi --speakers 2
```

Output: a readable transcript in the terminal + `data/<name>/transcript.json`
(turns, per-speaker talk stats, full text).

Options: `--model` (tiny|base|small|medium|large-v3), `--language` (auto if
omitted), `--speakers` (pin the count, e.g. 2 for a 1:1 call).

## Web UI

```bash
source env.sh
uvicorn backend.server:app --port 8000
```

Then open **http://localhost:8000** — upload a recording (drag & drop or click)
or paste a presigned/public S3 (or any https) URL, choose the model, optionally
set source language / speaker count / **translate-to** language, and click
**Transcribe**. This first pass does transcribe + diarize + translate only.
On the result, two buttons — **Summarize** and **Analyze** — run those stages on
demand (so you don't spend LLM time on every recording). Results render as:

- **Speaker-wise** turns (color-coded, with roles, + translation if requested)
- **Full text** (original + translated)
- **Summary** (outcome, sentiment, key points, action items)
- **QA analytics** (0–100 score + the standard scorecard, each checkpoint with
  score / verdict / evidence quote / suggestion)

CLI equivalents: `--translate <language>` and `--no-analytics`.

## Public Task API (for third-party integrations)

**Interactive docs:** open **http://localhost:8000/docs** (Swagger) or **/redoc** —
every endpoint, payload, and response is documented, with an **Authorize** button
to paste your `X-API-Key` and try calls live. Raw spec at **/openapi.json**.

Persistent, async, two-step. Enqueue a recording → get a `task_id` → poll →
then fetch transcript / run summarize / run analyze separately.

**Auth:** every `/api/v1/tasks` request needs the header `X-API-Key: <API_KEY>`
(set `API_KEY` in `.env`). Tasks + results are stored in SQLite (`data/tasks.db`)
and survive restarts.

| Method & path | Purpose |
|---|---|
| `POST /api/v1/tasks` | Enqueue (transcribe + diarize + translate). Returns `{ task_id, status:"queued" }` |
| `GET /api/v1/tasks` | List all tasks (processing + processed), newest first |
| `GET /api/v1/tasks/{id}` | Poll: status, stage, progress, `*_ready` flags |
| `GET /api/v1/tasks/{id}/transcript` | Speaker-wise turns + full text (+ translation) |
| `POST /api/v1/tasks/{id}/summarize` | Run summary (async) → poll, then GET summary |
| `GET /api/v1/tasks/{id}/summary` | `{ summary_status, summary }` |
| `POST /api/v1/tasks/{id}/analyze` | Run QA (async) → poll, then GET analytics |
| `GET /api/v1/tasks/{id}/analytics` | `{ analytics_status, analytics }` |
| `GET /api/v1/tasks/{id}/result` | Everything available so far, in one payload |

```bash
# 1. enqueue
curl -X POST http://HOST/api/v1/tasks -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"Audio Path":"https://…/call.wav","engine":"auto","language":"auto","translate_to_english":true}'
#    → { "task_id": "abc123", "status": "queued", "status_url": "/api/v1/tasks/abc123" }

# 2. poll until status == "done"
curl http://HOST/api/v1/tasks/abc123 -H "X-API-Key: $KEY"

# 3. speaker-wise + full text
curl http://HOST/api/v1/tasks/abc123/transcript -H "X-API-Key: $KEY"

# 4. summary (run, then poll summary_status → done, then read)
curl -X POST http://HOST/api/v1/tasks/abc123/summarize -H "X-API-Key: $KEY"
curl http://HOST/api/v1/tasks/abc123/summary -H "X-API-Key: $KEY"

# 5. QA analytics (same pattern)
curl -X POST http://HOST/api/v1/tasks/abc123/analyze -H "X-API-Key: $KEY"
curl http://HOST/api/v1/tasks/abc123/analytics -H "X-API-Key: $KEY"
```

Submit fields: `Audio Path` (or `audio_url`/`url`), `engine` (`auto`→large-v3),
`language` (`auto`→detect), `translate_to_english` (bool), `speakers` (int, optional),
`callback_url` (optional webhook, called when the transcribe pass finishes).

---

### Legacy all-in-one endpoint

`POST /api/v1/jobs` runs transcribe + summary + QA in one call (webhook or poll).
Kept for backward compatibility; new integrations should use `/api/v1/tasks`.

#### `POST /api/v1/jobs`

```json
{
  "Audio Path": "https://bucket.s3.amazonaws.com/call.wav",
  "callback_url": "https://your-server.example.com/webhook",
  "engine": "auto",
  "language": "auto",
  "translate_to_english": true,
  "speakers": null
}
```

| Field | Meaning |
|---|---|
| `Audio Path` | audio URL (presigned/public S3 or any https) — aliases: `audio_path`, `audio_url`, `url` |
| `callback_url` | optional; result is POSTed here when done. Omit to poll instead |
| `engine` | `"auto"` → `large-v3` (best). Or a size: `small`/`medium`/`large-v3`/`large-v3-turbo` |
| `language` | `"auto"` → auto-detect. Or a code (`hi`, `ml`, `en`, …) |
| `translate_to_english` | `true` → add English translation (skipped if already English) |
| `speakers` | optional int to pin the speaker count (e.g. `2`) |

Response (`202 Accepted`):
```json
{ "job_id": "fa7bdbfae9e6", "status": "queued", "status_url": "/api/v1/jobs/fa7bdbfae9e6",
  "message": "Processing started. Results will be POSTed to callback_url when ready." }
```

### `GET /api/v1/jobs/{job_id}` — poll status/result
Same JSON shape that is POSTed to the callback.

### Result payload (callback body / poll response when `status: "done"`)
```json
{
  "job_id": "…", "status": "done",
  "audio_path": "…", "language": "en", "translated_to": "English",
  "duration_seconds": 302, "processing_seconds": 194,
  "roles": {"agent": "Speaker 1", "customer": "Speaker 2"},
  "speakers": {"Speaker 1": {"talk_seconds": 234, "talk_share_pct": 77.6}, "…": {}},
  "speaker_wise": [
    {"speaker": "Speaker 1", "role": "agent", "start": 0.0, "end": 4.3,
     "text": "…", "text_translated": "…"}
  ],
  "full_text": "…",
  "full_text_translated": "…",
  "summary": {"summary": "…", "outcome": "Follow-up", "key_points": [],
              "action_items": [], "customer_sentiment": {}, "agent_tone": "…"},
  "analytics": {"overall_score": 40, "compliance_fail": true,
                "scorecard": [{"category": "…", "checkpoint": "…", "score": 4,
                               "verdict": "Met", "evidence": "… [00:24]", "suggestion": "…"}]}
}
```
On failure: `{ "job_id": "…", "status": "error", "error": "…" }`.
