# ASR Call Analytics Tool — Technical Approach

**Goal:** A local, zero-cost web tool that ingests a call recording and produces:
transcription (any language → any language), speaker-wise conversation, full
transcript, whole-conversation summary, and AI quality analytics on a standard
QA scorecard.

**Constraints locked in:**
- Runs locally on Apple Silicon Mac — **$0 cost**
- Web UI (upload a recording in the browser, view results)
- Local LLM via **Ollama** (nothing leaves the machine — fully private)

---

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  BROWSER (Web UI)                                            │
│  Upload recording · pick source/target language · view       │
│  transcript, speakers, summary, analytics · export report    │
└───────────────┬─────────────────────────────────────────────┘
                │ HTTP (localhost)
┌───────────────▼─────────────────────────────────────────────┐
│  BACKEND — FastAPI (Python)                                  │
│                                                             │
│  ① Preprocess   ffmpeg → 16kHz mono WAV                      │
│  ② ASR+Diarize  WhisperX (faster-whisper + pyannote)         │
│  ③ Translate    LLM (default) or NLLB-200                    │
│  ④ Assemble     merge speaker turns + text                   │
│  ⑤ Summarize    Ollama LLM                                   │
│  ⑥ Analyze      Ollama LLM → QA scorecard JSON               │
│                                                             │
│  Storage: SQLite (metadata + results) + local files         │
└─────────────────────────────────────────────────────────────┘
```

Processing runs as a **background job** (calls take seconds to minutes), and the
UI polls job status — so the browser never blocks.

---

## 2. Tech stack

| Layer | Choice | Why | Cost |
|---|---|---|---|
| Backend | **Python 3.11 + FastAPI + Uvicorn** | Async, all ML libs are Python | Free |
| Audio | **ffmpeg** | Any format → normalized WAV | Free |
| ASR + Diarization | **WhisperX** (`faster-whisper` backend + `pyannote.audio`) | Transcription + word timestamps + speaker labels in one tool; Metal/MPS accelerated on Apple Silicon | Free (local) |
| Translation | **Ollama LLM** (default) or **NLLB-200-distilled** | Any-to-any, 200 languages | Free (local) |
| LLM (summary + analytics) | **Ollama** running **Qwen2.5** (7B or 14B) | Strong multilingual + reasoning, JSON output | Free (local) |
| DB | **SQLite** | Zero-config, file-based | Free |
| Frontend | **Plain HTML + Alpine.js + Tailwind (CDN-free build)** or minimal React | Lightweight, served by FastAPI | Free |
| Job queue | In-process background tasks (or **RQ + SQLite**) | No external broker needed | Free |

**Model recommendation for Apple Silicon:**
- Whisper: `large-v3` if ≥16GB RAM, else `medium` (good multilingual accuracy).
- LLM: `qwen2.5:7b` on 16GB, `qwen2.5:14b` on 32GB+.

---

## 3. Pipeline — stage by stage

### ⓪ Ingest (inputs)
Two ways to provide a recording:
- **File upload** — any audio/video container; accepted by *probing with ffmpeg*, not by file extension.
- **URL** — any HTTPS link, including **presigned or public S3 URLs**. Downloaded over HTTP; no AWS credentials.
- *(Deferred)* private `s3://bucket/key` via boto3 + AWS keys.

### ① Preprocess
`ffmpeg -i input.<any> -ar 16000 -ac 1 -c:a pcm_s16le output.wav`
- Normalizes any format (mp3, m4a, opus, wav, aac, flac, ogg, opus, wma, amr, and audio-in-video mp4/mov/mkv) to 16kHz mono.
- Optional: loudness normalization + silence trim for cleaner ASR.

### ② ASR + Diarization (WhisperX)
1. **Transcribe** with faster-whisper → segments in the *original* language + word timestamps.
2. **Align** (wav2vec2) → precise word-level timestamps.
3. **Diarize** with pyannote → "SPEAKER_00 spoke 0.0–4.2s", etc.
4. **Assign** speakers to words → speaker-labeled transcript.

> Requires a **free HuggingFace token** to download the pyannote model (one-time,
> accept the model's license). No usage cost.

Output: list of turns `{speaker, start, end, text}` in the source language.

### ③ Translation (any → any)
Two modes, selectable per job:
- **LLM mode (default):** feed source turns to Qwen2.5, translate to target language, preserve speaker labels & timing. Fewer moving parts, context-aware.
- **NLLB mode (optional):** dedicated MT model for higher-fidelity literal translation of long calls.

If source == target, this stage is skipped.

### ④ Assemble
Merge consecutive same-speaker turns, attach timestamps, produce:
- `full_text` — clean running transcript
- `speaker_turns` — array of labeled turns (source + translated)
- `talk_stats` — per-speaker talk time, talk-to-listen ratio, interruptions, dead-air spans (derived from diarization timing)

**Speaker naming:** UI lets you rename `SPEAKER_00 → Agent`, `SPEAKER_01 → Customer`.

### ⑤ Summarize
Ollama LLM produces:
- 3–5 sentence executive summary
- Key points / bullet highlights
- Action items & follow-ups
- Call outcome (Resolved / Escalated / Follow-up / Unresolved)

### ⑥ Analyze — QA scorecard
LLM scores the standard checkpoints (see §5). Each returns:
```json
{
  "checkpoint": "Empathy & acknowledgment",
  "category": "Communication Skills",
  "score": 4,
  "verdict": "Met",
  "evidence": "Agent: 'I understand how frustrating that is' @ 02:14",
  "suggestion": "Acknowledge the delay earlier"
}
```
- **Evidence field quotes the real transcript line + timestamp** → auditable, trustworthy scores.
- **Compliance failures** apply heavy penalty / auto-flag rather than simple averaging.
- Rolls up to a weighted **0–100 overall QA score**.

---

## 4. Data model (SQLite)

```
recordings(id, filename, path, duration, source_lang, target_lang,
           status, created_at)
transcripts(recording_id, full_text_src, full_text_tgt, turns_json)
analytics(recording_id, summary_json, scorecard_json, talk_stats_json,
          overall_score, outcome)
speakers(recording_id, speaker_id, display_name)
```
Audio + intermediate JSON stored under `data/<recording_id>/`.

---

## 5. QA scorecard (the standard set — locked)

**Opening & Professionalism:** greeting & self-ID · professional tone · active listening
**Communication:** clarity · empathy · interruptions · dead air / hold
**Problem Handling:** issue identification · knowledge accuracy · resolution · efficiency
**Compliance:** mandatory disclosures · verification · data privacy *(auto-fail weighting)*
**Closing:** summary & next steps · additional help offered · proper close
**Derived metrics:** customer sentiment trend · agent tone · talk-to-listen ratio · call outcome · overall 0–100 score

---

## 6. API (FastAPI endpoints)

```
POST /api/recordings         upload file + source/target lang → returns job id
GET  /api/recordings         list all
GET  /api/recordings/{id}    status + full results
POST /api/recordings/{id}/speakers   rename speakers
GET  /api/recordings/{id}/export     download report (HTML/PDF/JSON)
DELETE /api/recordings/{id}
```

---

## 7. Web UI (screens)

1. **Upload** — drag file, choose source language (or auto-detect) + target language, model size.
2. **Processing** — live status per stage (transcribing → diarizing → translating → analyzing).
3. **Results**
   - *Overview*: summary, outcome, overall QA score gauge, talk-to-listen donut.
   - *Transcript*: toggle original / translated; speaker-colored turns with timestamps; click a turn to jump.
   - *Analytics*: scorecard by category, each checkpoint with score, verdict, evidence quote, suggestion.
   - *Export*: one-click HTML/PDF report.
4. **Library** — all processed calls, searchable, with scores.

---

## 8. Project structure

```
asr/
├── backend/
│   ├── main.py               # FastAPI app + routes
│   ├── pipeline/
│   │   ├── preprocess.py      # ffmpeg
│   │   ├── transcribe.py      # WhisperX ASR + diarization
│   │   ├── translate.py       # LLM / NLLB
│   │   ├── assemble.py        # turns + talk stats
│   │   ├── summarize.py       # Ollama summary
│   │   └── analyze.py         # QA scorecard
│   ├── llm.py                 # Ollama client + prompts
│   ├── db.py                  # SQLite models
│   └── jobs.py                # background job runner
├── frontend/                  # UI (static, served by FastAPI)
├── data/                      # audio + results (gitignored)
├── requirements.txt
└── README.md
```

---

## 9. Performance & sizing (Apple Silicon)

| Model | RAM | ~Speed (per min of audio) |
|---|---|---|
| Whisper medium | 8–16GB | ~0.3–0.6× realtime |
| Whisper large-v3 | 16GB+ | ~0.6–1× realtime |
| pyannote diarization | +2GB | ~0.2× realtime |
| Qwen2.5 7B (summary+analytics) | ~8GB | seconds per call |

A typical 5-minute call ≈ **2–5 minutes** end-to-end on an M-series with 16GB.
Models are downloaded once and cached; subsequent runs are offline.

---

## 10. One-time setup (prerequisites)

1. `brew install ffmpeg`
2. Python venv + `pip install -r requirements.txt` (whisperx, faster-whisper, pyannote.audio, fastapi, uvicorn, ollama)
3. Install **Ollama** (`brew install ollama`) → `ollama pull qwen2.5:7b`
4. Free **HuggingFace account** → token → accept pyannote model license (one-time)

All downloads are free; after setup the tool runs fully offline.

---

## 11. Cost summary

| Item | Cost |
|---|---|
| Whisper / WhisperX | $0 (open source, local) |
| pyannote diarization | $0 (free model, HF token) |
| Translation (LLM/NLLB) | $0 (local) |
| Ollama LLM | $0 (local) |
| FastAPI + SQLite + UI | $0 |
| **Total** | **$0 — only your electricity** |

---

## 12. Build roadmap

- **Phase 1 — Core pipeline ✅ DONE:** ffmpeg → faster-whisper → pyannote (community-1) → speaker-wise turns + talk stats. CLI + web, verified.
- **Phase 2 — LLM stages ✅ DONE:** translation + summary (roles/outcome/sentiment/action items) + QA scorecard via Ollama/qwen2.5. Verified end-to-end in the web UI.
- **Phase 3 — API + jobs ✅ (in-memory):** FastAPI endpoints + background job runner done. SQLite persistence still TODO (jobs are currently in-memory).
- **Phase 4 — Web UI ✅ DONE:** upload/URL, live progress, speaker-wise/full/summary/analytics tabs. Export (PDF) still TODO.
- **Phase 5 — Polish (TODO):** SQLite library + search, speaker renaming, PDF/HTML export, batch, transferred-call stitching.

Build the pipeline first as a script, verify quality on your own recordings,
*then* wrap it in the web UI. That de-risks the hard part (ASR accuracy) before
investing in the interface.

### Deferred / backlog (agreed, not in initial build)
- **Transferred-call stitching** — when a call is transferred, the system creates
  a 2nd recording (Customer + new Agent). Stitch the two sequential legs into one
  interaction (Customer continuous; Agent A → Agent B), max two files. Each leg
  runs the normal pipeline; add a concatenation + transfer-marker step and
  per-agent vs overall analytics. Ordering by filename timestamp or upload order.
- **Private `s3://` ingestion** via boto3 + AWS credentials.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Diarization struggles on overlapping speech / phone audio | Use 16kHz clean mono; pyannote handles 2-speaker calls well; expose min/max speaker hints |
| Whisper mistranslates rare languages | Transcribe in source lang first, translate separately; let user correct source language |
| LLM returns invalid JSON for scorecard | Enforce JSON schema + retry; use structured output |
| Large models slow on 8GB Macs | Fall back to Whisper `medium` + `qwen2.5:3b`; process in background |
| Long calls exceed LLM context | Chunk transcript, map-reduce summary/analytics |
```
