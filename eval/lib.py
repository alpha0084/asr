"""Shared helpers for the transcription eval harness.

Runs ASR only (no diarization/LLM) so we measure word accuracy fast, and scores
hypotheses against hand-corrected references with WER/CER.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))          # allow `import backend...`

import jiwer

from backend.config import AUDIO_FILTERS, DATA_DIR
from backend.pipeline import ingest, preprocess, transcribe

REFS = ROOT / "refs"
OUT = ROOT / "out"
DATASET = ROOT / "dataset.json"
EVAL_WORK = DATA_DIR / "_eval"

# Normalize both sides before scoring: lowercase, strip punctuation, collapse spaces.
_clean = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def load_dataset() -> list:
    if not DATASET.exists():
        return []
    return json.loads(DATASET.read_text())


def transcribe_text(entry: dict, model: str, language: str | None = None,
                    clean: bool = False) -> tuple[str, str]:
    """ASR-only transcription of one dataset entry → (full text, detected language)."""
    src = entry["audio"]
    lang = language if language is not None else entry.get("language")
    if lang in ("", "auto", None):
        lang = None

    path = ingest.resolve_input(src)
    EVAL_WORK.mkdir(parents=True, exist_ok=True)
    suffix = "_clean" if clean else ""
    wav = preprocess.to_wav(path, EVAL_WORK / f"{path.stem}{suffix}.wav",
                            clean=clean, filters=AUDIO_FILTERS)
    waveform, sr = preprocess.load_waveform(wav)

    tr = transcribe.transcribe(waveform, sr, model_name=model, language=lang)
    text = " ".join(s["text"].strip() for s in tr["segments"]).strip()
    return text, tr.get("language") or (lang or "?")


def normalize(s: str) -> str:
    return _clean(s or "")


def score(ref: str, hyp: str) -> tuple[float, float]:
    """Return (WER, CER) after normalization."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return (0.0, 0.0) if not h else (1.0, 1.0)
    return jiwer.wer(r, h), jiwer.cer(r, h)
