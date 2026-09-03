"""Stage 2 — speech-to-text with word timestamps.

Three backends:
  - "mlx"    : Whisper on the Apple GPU via mlx-whisper (fast; default on Apple Silicon)
  - "cuda"   : faster-whisper on an NVIDIA GPU (fast; for the RTX box)
  - "faster" : faster-whisper on CPU (reliable fallback)
The "cuda"/"faster" backends share one code path (CTranslate2) differing only by
device; all three return {language, language_probability, segments:[{start,end,text,words}]}.
"""
import os

from .. import settings

_faster_cache = {}

# Map friendly model names -> mlx-community HF repos (a raw repo id passes through).
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def transcribe(waveform, sr, model_name=None, language=None, on_progress=None, backend=None):
    """Transcribe a 16kHz mono float32 waveform.

    language=None -> auto-detect. Tries the configured backend, falls back to
    faster-whisper (CPU) if MLX/CUDA is unavailable or errors.
    """
    assert sr == 16000, "waveform must be 16kHz (preprocess first)"
    backend = (backend or settings.asr_backend()).lower()
    name = model_name or settings.whisper_model()

    if backend == "mlx":
        try:
            return _transcribe_mlx(waveform, sr, name, language, on_progress)
        except Exception as e:
            print(f"[transcribe] MLX unavailable ({type(e).__name__}: {e}); "
                  f"falling back to faster-whisper (CPU)")
            backend = "faster"

    if backend == "cuda":
        try:
            return _transcribe_faster(waveform, sr, name, language, on_progress, device="cuda")
        except Exception as e:
            print(f"[transcribe] CUDA unavailable ({type(e).__name__}: {e}); "
                  f"falling back to faster-whisper (CPU)")
            backend = "faster"

    return _transcribe_faster(waveform, sr, name, language, on_progress, device="cpu")


# ---------------------------------------------------------------- MLX (GPU)
def _transcribe_mlx(waveform, sr, name, language, on_progress):
    import mlx_whisper

    repo = MLX_REPOS.get(name, name)
    result = mlx_whisper.transcribe(
        waveform,
        path_or_hf_repo=repo,
        language=language,
        word_timestamps=True,
        verbose=None,
    )
    segs = []
    raw = result.get("segments", [])
    total = (raw[-1]["end"] if raw else 0) or (len(waveform) / sr)
    for s in raw:
        if on_progress and total:
            on_progress(min(1.0, s["end"] / total), s["end"], total)
        segs.append({
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip(),
            "words": [
                {"start": w["start"], "end": w["end"], "word": w["word"]}
                for w in s.get("words", [])
            ],
        })
    return {
        "language": result.get("language"),
        "language_probability": None,
        "segments": segs,
    }


# ------------------------------------------------------- faster-whisper (CPU / CUDA)
def _get_faster(name: str, device: str):
    # int8 is the sane default on CPU; float16 uses the NVIDIA tensor cores on CUDA.
    # WHISPER_COMPUTE_TYPE overrides both (e.g. int8_float16 for a smaller CUDA footprint).
    key = (name, device)
    if key not in _faster_cache:
        from faster_whisper import WhisperModel
        compute_type = (os.getenv("WHISPER_COMPUTE_TYPE", "").strip()
                        or ("float16" if device == "cuda" else "int8"))
        kwargs = {"device": device, "compute_type": compute_type}
        if device == "cpu":
            kwargs["cpu_threads"] = int(os.getenv("WHISPER_THREADS", "8"))
        _faster_cache[key] = WhisperModel(name, **kwargs)
    return _faster_cache[key]


def _transcribe_faster(waveform, sr, name, language, on_progress, device="cpu"):
    # MLX repo ids won't load here; map back to a plain size if needed.
    if name not in MLX_REPOS and name.startswith("mlx-community/"):
        name = "large-v3"
    model = _get_faster(name, device)
    segments, info = model.transcribe(
        waveform, language=language, task="transcribe",
        word_timestamps=True, vad_filter=True, beam_size=5,
    )
    total = getattr(info, "duration", 0) or (len(waveform) / sr)
    segs = []
    for s in segments:
        if on_progress and total:
            on_progress(min(1.0, s.end / total), s.end, total)
        segs.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "words": [
                {"start": w.start, "end": w.end, "word": w.word}
                for w in (s.words or [])
            ],
        })
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "segments": segs,
    }
