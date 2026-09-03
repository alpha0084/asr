# Changelog

All notable changes to the ASR Call Analytics tool. Newest first.

> **Rollback:** this project is not under git. To roll back a change, restore the
> corresponding files from `backups/<label>/` (each entry below names its backup).

---

## [0.7.0] — Admin portal — 2026-09-01
Single-admin web portal at **`/admin`** to run the platform without touching .env/code.
- **Auth:** one admin password (pbkdf2-sha256, stdlib), seeded from `ADMIN_PASSWORD`
  on first run; in-process bearer/cookie sessions (`admin_auth.py`).
- **Dynamic config** (persisted in SQLite, overlaying .env via `settings.py`):
  QA scorecard categories/checkpoints/**weights** (`analyze.py` now data-driven,
  weighted rollup), summary fields (outcome options, sentiment toggle, custom
  fields), models/backend (`WHISPER_MODEL`/`ASR_BACKEND`/`OLLAMA_MODEL`/target lang),
  and create/revoke of X-API-Keys (`require_api_key` accepts any active key).
- **Global webhook** (`webhooks.py`): fires `transcribed`/`summarized`/`analyzed`
  events to a configured URL with retries + exponential backoff, HMAC-SHA256
  signature (`X-ASR-Signature`), and a delivery log; per-request `callback_url`
  still works. Resumes pending deliveries on restart.
- **Dashboard:** all recordings + statuses, queue monitor, and actions — re-run
  analyze/summarize, re-transcribe, re-fire a failed webhook, delete (with file
  cleanup). New tables `settings`, `api_keys`, `webhook_events` in `db.py`.
- New: `settings.py`, `admin_auth.py`, `admin_api.py`, `webhooks.py`,
  `frontend/admin.html`. **Backup:** `backups/pre-admin-portal/`.

## [0.6.5] — CUDA transcription backend — 2026-09-01
- New `ASR_BACKEND=cuda`: runs faster-whisper on an NVIDIA GPU (RTX box) via
  CTranslate2 — same code path as the CPU `faster` backend, differing only by
  `device`. `float16` by default on CUDA, `int8` on CPU; `WHISPER_COMPUTE_TYPE`
  overrides. `mlx` and `cuda` both fall back to CPU faster-whisper on failure.
- `_get_faster` now keys its model cache by `(name, device)` so CPU/GPU models
  don't collide. Config comment + `.env.example` updated. **Backup:** `backups/pre-cuda-backend/`.

## [0.6.4] — Reverted speaker experiments + audio cleanup — 2026-08-24
Rolled back 0.6.1 / 0.6.2 / 0.6.3 (LLM diarization, audio cleanup, hybrid) — none
were reliably better on mono/8kHz audio. Restored `runner.py`, `config.py`,
`assemble.py` from `backups/pre-llm-diarize/`: plain pyannote segment-level
diarization, roles inferred by the summary. `refine_speakers.py` remains on disk
but unused. Speaker accuracy on mono audio remains blocked on dual-channel
recordings / cleaner transcription.

## [0.6.3] — Hybrid speaker assignment (pyannote + LLM orientation) — 2026-08-24 ⟲ REVERTED
- New default `DIARIZE_METHOD=hybrid`: pyannote finds turn boundaries, then ONE LLM
  call (`refine_speakers.orient`) decides which cluster is the Agent vs Customer.
  Reliable role identity (fixes the agent/customer swapping) since it's a single
  global decision, not per-line labeling. Some pyannote merging remains (accepted).
- `assemble.assign_speakers` snaps unmatched segments to the nearest speaker (no
  stray SPEAKER_? cluster).
- UI: don't duplicate the role when the speaker label already is the role (was
  showing "Customer (Customer)").
- `llm` and `pyannote` methods still available via `DIARIZE_METHOD`.

## [0.6.2] — Phase A: audio cleanup before ASR — 2026-08-24
- `preprocess.to_wav(clean=, filters=)` applies an ffmpeg cleanup chain (high-pass
  + FFT denoise + dynamic normalization) before ASR — denoise + lift a quiet
  speaker. No new deps (ffmpeg only).
- Config `AUDIO_CLEANUP` (default off) + `AUDIO_FILTERS`; wired into
  `runner.transcribe_only` and the eval harness (`score.py --clean`).
- Early qualitative signal (c22 clip): cleaned audio yields markedly more coherent
  text (correct question forms, +4 dB on quiet speech). Sample clips in
  `eval/samples/`. Enable by default only once WER numbers confirm it.

## [0.6.1] — LLM-based speaker assignment (mono audio) — 2026-08-24
- New `backend/pipeline/refine_speakers.py`: splits the transcript into
  sentence-level units (anchored to Whisper word timestamps) and has qwen2.5 label
  each **agent/customer** by conversational role — fixes the "merged turns" problem
  on mono/8kHz audio where acoustic diarization (pyannote) can't separate voices.
- `DIARIZE_METHOD` config toggle (`llm` default | `pyannote`); LLM path skips
  pyannote entirely and falls back to it on failure. `runner.transcribe_only`
  branches on it; `assemble.talk_stats_from_turns` added; `llm.chat*` take options.
- Known limitation: role labels can still swap on garbled transcripts — cleaner
  transcription (0.6.0 eval + Phase A) will improve it. **Backup:** `backups/pre-llm-diarize/`.

## [0.6.0] — Transcription accuracy eval harness — 2026-08-24
- New `eval/` harness to measure transcription **WER/CER** (via `jiwer`), so
  accuracy changes are judged by numbers, not eyeballing (Step 0 of the accuracy plan).
- `eval/dataset.json` (test set), `eval/make_refs.py` (draft transcripts to
  hand-correct), `eval/score.py` (per-call / per-language / overall WER+CER),
  `eval/lib.py` (ASR-only transcription + normalized scoring), `eval/README.md`.
- Verified end-to-end (0% WER on a matched sample). `eval/out/` gitignored;
  `eval/refs/` (ground truth) kept.

## [0.5.1] — Swagger / OpenAPI documentation — 2026-08-20
- Full interactive docs at **`/docs`** (Swagger), **`/redoc`**, **`/openapi.json`**.
- `backend/schemas.py`: typed request/response models → every payload & response
  has a documented schema (17 models).
- `X-API-Key` documented as a security scheme (Swagger **Authorize** button); each
  protected endpoint marked as requiring it.
- Rich per-endpoint summaries/descriptions, error responses (401/404/409), tags
  (Tasks API / Web UI / Legacy), and a top-level flow guide.
- **Backup:** `backups/pre-swagger/server.py`

## [0.5.0] — Third-party Task API (persistent, async, API-key) — 2026-08-20
- New **task API** under `/api/v1/tasks` mirroring the two-step flow:
  `POST /tasks` (enqueue, transcribe-only) → `GET /tasks/{id}` (poll) →
  `GET /tasks/{id}/transcript` · `POST /tasks/{id}/summarize` + `GET …/summary` ·
  `POST /tasks/{id}/analyze` + `GET …/analytics` · `GET /tasks` (list all).
- **SQLite persistence** (`backend/db.py`, `data/tasks.db`): tasks + results survive
  restarts; unfinished tasks are marked `interrupted` on boot.
- **API-key auth**: all `/api/v1/tasks` endpoints require `X-API-Key` (set `API_KEY`
  in `.env`). Web UI endpoints (`/api/jobs`) remain local/unauthenticated.
- Summarize/Analyze are **async** (enqueue + poll their `*_status`).
- **Backup of pre-change files:** `backups/pre-task-api/` (server.py, jobs.py,
  config.py, public.py)

---

## [0.4.0] — Two-step web flow (transcribe → summarize/analyze on demand) — 2026-08-20
**Goal:** don't spend LLM time on summary/QA for every recording.
- Web UI now runs **transcribe + diarize + translate** first (speaker-wise + full
  text only). **Summarize** and **Analyze** are separate buttons on the result,
  each runs on demand and shows its own progress.
- `runner.py` split into `transcribe_only()` / `add_summary()` / `add_analytics()`
  / `process()` (all-in-one).
- `summarize.infer_roles()` added so **Analyze** works even without Summarize first.
- New endpoints: `POST /api/jobs/{id}/summarize`, `POST /api/jobs/{id}/analyze`;
  job tracks `summary_status` / `analytics_status`.
- Public API v1 unchanged for callers (still all-in-one). Fixed: the all-in-one
  path now stays `running` until summary+QA finish (previously reported `done`
  after transcription).
- **Backup of pre-change files:** `backups/pre-two-step/` (runner.py, jobs.py,
  server.py, index.html)

---

## [0.3.0] — Word-level speaker assignment — 2026-08-19  ⟲ ROLLED BACK
**Status:** reverted to 0.2.0 (segment-level) on 2026-08-19 — residual diarization
errors on mono/8kHz audio remained, so the change wasn't worth keeping. Files
restored from `backups/pre-word-level/`. Kept here for reference.

**Goal:** fix customer speech being mislabeled as the agent on mono calls.
- `assemble.py` now assigns speakers at the **word** level (using Whisper word
  timestamps) and splits turns at speaker boundaries, instead of labeling a whole
  Whisper segment as one speaker. Removes the `SPEAKER_?` gaps (words snap to the
  nearest speaker) and smooths sub-0.45s speaker "blips".
- `runner.py` calls the new `assemble.assign_and_build()`.
- **Improves** boundary cases and removes unknown-speaker gaps. Residual errors on
  hard mono/8kHz audio (quiet/overlapping speech) are limited by diarization
  accuracy itself, not the assignment step.
- **Rollback:** restore `backups/pre-word-level/assemble.py` and `runner.py`, then
  restart the server.

---

## [0.2.0] — Rollback checkpoint (before word-level diarization) — 2026-08-19

This is the fully-working state *before* the word-level change. Restore
`backups/pre-word-level/assemble.py` and `runner.py` to return here.

**Environment (Apple M5 / 16GB, all local, $0)**
- ffmpeg 9 (+ ffmpeg@7), Python 3.11 venv, Ollama + qwen2.5:7b
- Whisper via **mlx-whisper (Apple GPU)** with faster-whisper (CPU) fallback
- pyannote `speaker-diarization-community-1` on MPS; audio loaded in-memory via
  soundfile (no torchcodec)

**Pipeline** (`backend/pipeline/`)
- `ingest` — file path or http(s)/presigned-S3 URL
- `preprocess` — ffmpeg → 16kHz mono WAV → soundfile waveform
- `transcribe` — MLX GPU (large-v3 ~7× realtime) + CPU fallback, word timestamps
- `diarize` — pyannote community-1, in-memory waveform, optional pinned speakers
- `assemble` — **segment-level** speaker assignment + talk stats
- `translate` — LLM, any→any, auto-skips when source == target language
- `summarize` — summary, outcome, sentiment, roles, action items (Ollama)
- `analyze` — QA scorecard (17 checkpoints, 0–100, evidence-backed, compliance fail)

**Interfaces**
- CLI: `python -m backend.cli <file/url> [--model --language --speakers --translate --no-analytics]`
- Web UI: upload/URL, live per-stage progress, tabs (speaker-wise / full / summary / QA)
- Public API v1: `POST /api/v1/jobs` (async; webhook callback **or** polling),
  `GET /api/v1/jobs/{id}`; documented in README, sample in `examples/`

**Key fixes along the way**
- MLX GPU transcription (large-v3) — fixed slow CPU transcription and Malayalam
  mis-detection that `small` produced
- Auto-detect source language + translate-to-target; skip self-translation
- Live progress for Transcribe / Translate / Analyze stages
- Diarization on MPS (GPU)
- Reliable server restart (tracked background + kill-by-PID)
