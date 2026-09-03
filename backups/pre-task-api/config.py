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

# ASR backend: "mlx" runs Whisper on the Apple GPU (fast); "faster" is CPU fallback.
ASR_BACKEND = os.getenv("ASR_BACKEND", "mlx").strip()

# pyannote uses torch and CAN use MPS; CPU is the reliable default. Override
# with DIARIZE_DEVICE=mps to use Metal.
DIARIZE_DEVICE = os.getenv("DIARIZE_DEVICE", "cpu").strip()

# --- Paths ---
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR = DATA_DIR / "_downloads"
