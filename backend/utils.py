"""Small shared helpers."""
import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stderr():
    """Silence C-level stderr at the file-descriptor level.

    pyannote pulls in torchcodec, which prints a noisy (harmless) ffmpeg-link
    traceback on import because Homebrew ships ffmpeg 9. We never use torchcodec
    (audio is loaded via soundfile), so we just mute that output.
    """
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def fmt_ts(seconds: float) -> str:
    """Seconds -> mm:ss (or h:mm:ss for long calls)."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcript_text(turns: list, key: str = "text") -> str:
    """Render turns as '[mm:ss] Speaker: text' lines for LLM input."""
    return "\n".join(
        f"[{fmt_ts(t['start'])}] {t['speaker']}: {t.get(key) or t['text']}"
        for t in turns
    )


def speaker_label(raw: str, mapping: dict) -> str:
    """Map pyannote's SPEAKER_00 -> friendly 'Speaker 1' (stable per call)."""
    if raw not in mapping:
        mapping[raw] = f"Speaker {len(mapping) + 1}"
    return mapping[raw]
