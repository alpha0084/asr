#!/usr/bin/env bash
# Source this before running the tool:  source env.sh
# Activates the Python 3.11 venv for the ASR tool.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/.venv/bin/activate"

echo "ASR env ready — Python $(python --version 2>&1 | awk '{print $2}')"

# --- Audio decoding note ---
# We do NOT use torchcodec (pyannote's default audio loader). Homebrew ships
# ffmpeg 9, which torchcodec can't link (it supports ffmpeg 4-7), and forcing
# ffmpeg@7's libs onto the path collides with PyAV's bundled ffmpeg -> crashes.
# Instead the pipeline preprocesses audio to 16kHz mono WAV (ffmpeg 9 CLI) and
# loads it in-memory with soundfile, handing waveforms to Whisper and pyannote.
# => No torchcodec, no DYLD_LIBRARY_PATH override, no library conflict.
