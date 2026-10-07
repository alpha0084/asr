"""Client for the warm Sortformer diarization microservice (runs in the isolated .venv-nemo).

Keeps NeMo + its heavy deps OUT of the prod venv: we just POST the wav path over a local socket and
get speaker segments back. The service holds the model warm (~1 GB VRAM), so a call is ~1 s.
"""
import json
import os
import urllib.request

SORTFORMER_URL = os.getenv("SORTFORMER_URL", "http://127.0.0.1:5531/diarize")


def diarize(audio_path: str, timeout: float = 30.0) -> list:
    """Return [{start, end, speaker}] from the Sortformer service, sorted by start.

    Raises on any failure (service down, bad response, no segments) so the caller can fall back to
    pyannote — Sortformer is an upgrade, never a hard dependency of the pipeline.
    """
    body = json.dumps({"audio_path": audio_path}).encode()
    req = urllib.request.Request(
        SORTFORMER_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    segs = out.get("segments", [])
    if not segs:
        raise RuntimeError(out.get("error") or "sortformer returned no segments")
    turns = [
        {"start": float(s["start"]), "end": float(s["end"]), "speaker": s["speaker"]} for s in segs
    ]
    turns.sort(key=lambda t: t["start"])
    return turns
