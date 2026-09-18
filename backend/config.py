"""Central config — loads .env and exposes settings + paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Secrets / models ---
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()

# Diarization model (gated on HuggingFace — accept its license once, see README).
# community-1 is the native model for pyannote.audio 4.x (self-contained).
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# ASR backend: "mlx" = Apple GPU (fast), "cuda" = NVIDIA GPU (fast), "faster" = CPU.
# mlx/cuda both fall back to CPU faster-whisper if their accelerator is unavailable.
ASR_BACKEND = os.getenv("ASR_BACKEND", "mlx").strip()

# pyannote uses torch and CAN use MPS; CPU is the reliable default. Override
# with DIARIZE_DEVICE=mps to use Metal.
DIARIZE_DEVICE = os.getenv("DIARIZE_DEVICE", "cpu").strip()

# Speaker separation strategy:
#   "hybrid"   -> acoustic word-level turns (pyannote) PLUS one global LLM call to
#                 name the two clusters Agent/Customer. Clean turn grouping without
#                 per-line wobble, and roles populated. Recommended default.
#   "llm"      -> split the transcript into sentences (word-timestamp anchored) and
#                 label each line agent/customer with the LLM. Finest splits but the
#                 per-line labels can wobble on garbled/small-talk audio.
#   "acoustic" -> pyannote + word-level overlap only; generic Speaker 1/2, no roles.
# hybrid/llm fall back to acoustic automatically if the LLM step errors.
SPEAKER_MODE = os.getenv("SPEAKER_MODE", "hybrid").strip().lower()

# LLM speaker mode: also break an utterance when the silence gap between two
# consecutive words exceeds this many seconds — not just at sentence punctuation.
# This splits run-on ASR output (common on 8kHz mono) at natural turn pauses so
# different speakers stop merging. Lower = more splits (0.4–0.8 is sensible).
SPEAKER_PAUSE_GAP = float(os.getenv("SPEAKER_PAUSE_GAP", "0.6"))

# --- Audio cleanup (pre-ASR) ---
# Runs an ffmpeg filter chain before resampling to help Whisper on noisy/quiet
# phone audio. Default chain: drop sub-100Hz rumble, FFT denoise, then dynamic
# loudness-normalise so quiet speech is audible. Override the chain via
# AUDIO_CLEANUP_FILTERS; disable entirely with AUDIO_CLEANUP=false.
AUDIO_CLEANUP = os.getenv("AUDIO_CLEANUP", "true").strip().lower() in {"1", "true", "yes", "on"}
AUDIO_CLEANUP_FILTERS = os.getenv(
    "AUDIO_CLEANUP_FILTERS",
    "highpass=f=100,afftdn=nf=-25,dynaudnorm=f=200:g=15",
).strip()

# --- Public API ---
# Third parties must send this in the `X-API-Key` header. Set in .env.
API_KEY = os.getenv("API_KEY", "").strip()

# --- Admin portal ---
# First admin seeded on boot from ADMIN_EMAIL + ADMIN_PASSWORD (if no admins exist
# yet). After that, admins are managed in the DB (admin_users table) and more can be
# added from the portal — the portal supports multiple email/password logins.
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@asr.local").strip().lower()

# --- Global webhook default (admin can still change it live; the live value is
# persisted in the DB. These .env values are the fallback default so the webhook
# stays configured even if settings are ever reset). ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
WEBHOOK_ENABLED = os.getenv("WEBHOOK_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

# --- Database (dedicated PostgreSQL instance, isolated from production) ---
# Loopback-only Postgres on port 5433 (see data/pgdata). Every transcribe/
# summarize/analyze run is stored as a versioned row with the params used, plus
# normalized turns/scorecard tables for SQL analytics + accuracy verification.
DATABASE_URL = os.getenv("DATABASE_URL", "host=127.0.0.1 port=5433 dbname=asr user=asr_admin")

# Worker pool: how many recordings to process in parallel. The queue is durable
# (Postgres), so workers on other machines can pull from the same DB later — just
# point them at the same DATABASE_URL. Sized to GPU capacity; raise as GPUs are added.
# This is the BOOT default; the live value is set from the DB (settings.worker_concurrency)
# and can be changed at runtime from the admin panel (jobs.set_concurrency).
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "3"))

# Per-worker resource estimates for the admin concurrency headroom check. The Whisper /
# pyannote models are loaded ONCE and shared across worker threads, so a worker's marginal
# cost is the per-inference activation (not a full model copy) plus its audio buffers —
# hence the modest per-worker figures. Conservative defaults; override via env if profiling
# says otherwise. RESERVE keeps VRAM/RAM free for the co-located production stack + safety.
WORKER_VRAM_MB = int(os.getenv("WORKER_VRAM_MB", "2200"))      # est. VRAM per concurrent job
WORKER_RAM_GB = float(os.getenv("WORKER_RAM_GB", "1.5"))       # est. system RAM per concurrent job
VRAM_RESERVE_MB = int(os.getenv("VRAM_RESERVE_MB", "3000"))    # keep free for prod + headroom
RAM_RESERVE_GB = float(os.getenv("RAM_RESERVE_GB", "2"))       # keep free for the OS + prod
MAX_WORKER_CONCURRENCY = int(os.getenv("MAX_WORKER_CONCURRENCY", "12"))  # absolute hard cap

# Autoscaling: when a backlog of recordings builds, temporarily raise the live worker count above
# the admin-set resting baseline toward the backlog size — bounded by AUTOSCALE_MAX and the
# real-time VRAM/RAM headroom so a burst never overcommits the shared GPU — then settle back to the
# baseline after the queue has been idle for the cooldown. Toggleable live from the admin panel.
AUTOSCALE_ENABLED_DEFAULT = os.getenv("AUTOSCALE_ENABLED", "true").lower() not in ("0", "false", "no")
AUTOSCALE_MAX = int(os.getenv("AUTOSCALE_MAX", str(MAX_WORKER_CONCURRENCY)))   # burst ceiling
AUTOSCALE_INTERVAL_SEC = float(os.getenv("AUTOSCALE_INTERVAL_SEC", "5"))       # poll cadence
AUTOSCALE_COOLDOWN_SEC = float(os.getenv("AUTOSCALE_COOLDOWN_SEC", "45"))      # idle before scale-down
AUTOSCALE_STEP = int(os.getenv("AUTOSCALE_STEP", "2"))                         # scale-down granularity

# --- Paths ---
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR = DATA_DIR / "_downloads"
DB_PATH = str(DATA_DIR / "tasks.db")   # legacy SQLite (kept for reference/migration)
