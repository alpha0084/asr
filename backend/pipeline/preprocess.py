"""Stage 1 — normalize any audio/video to 16kHz mono, load as an in-memory waveform."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

TARGET_SR = 16000


def probe(path: Path) -> dict:
    """Return {duration, format, codec} using ffprobe. Also validates decodability."""
    out = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise ValueError(f"ffprobe could not read the file (unsupported/corrupt): {path}")
    info = json.loads(out.stdout or "{}")
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise ValueError(f"No audio stream found in: {path}")
    return {
        "duration": float(info.get("format", {}).get("duration", 0.0)),
        "format": info.get("format", {}).get("format_name", "?"),
        "codec": audio.get("codec_name", "?"),
    }


def to_wav(src: Path, out: Path, clean: bool = False, filters: str = "") -> Path:
    """Transcode `src` (any format, incl. audio-in-video) to 16kHz mono PCM WAV.

    clean=True applies an ffmpeg cleanup chain (`filters`) before resampling —
    denoise + boost quiet speech — to help ASR on noisy/quiet phone audio.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-i", str(src)]
    if clean and filters:
        cmd += ["-af", filters]
    cmd += ["-ar", str(TARGET_SR), "-ac", "1", "-c:a", "pcm_s16le", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-800:]}")
    return out


def load_waveform(wav: Path):
    """Load a WAV as float32 mono numpy array via soundfile (no ffmpeg/torchcodec)."""
    data, sr = sf.read(str(wav), dtype="float32")
    if data.ndim > 1:                       # safety: collapse to mono
        data = data.mean(axis=1)
    return np.ascontiguousarray(data), sr
