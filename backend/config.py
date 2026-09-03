"""Central config — loads .env and exposes settings + paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Secrets / models ---
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3").strip()
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

# --- Database (dedicated PostgreSQL instance, isolated from production) ---
# Loopback-only Postgres on port 5433 (see data/pgdata). Every transcribe/
# summarize/analyze run is stored as a versioned row with the params used, plus
# normalized turns/scorecard tables for SQL analytics + accuracy verification.
DATABASE_URL = os.getenv("DATABASE_URL", "host=127.0.0.1 port=5433 dbname=asr user=asr_admin")

# --- Paths ---
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR = DATA_DIR / "_downloads"
DB_PATH = str(DATA_DIR / "tasks.db")   # legacy SQLite (kept for reference/migration)
