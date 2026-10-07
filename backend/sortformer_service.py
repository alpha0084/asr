"""Warm Sortformer diarization microservice.

Loads nvidia/diar_sortformer_4spk-v1 ONCE and serves diarization over a local HTTP socket, so the prod
pipeline (a different venv, no NeMo) can use it via `pipeline/sortformer_client.py`. Run with the
ISOLATED NeMo python (.venv-nemo). Binds 127.0.0.1:5531.

Long audio is diarized in overlapping WINDOWS. The model runs full self-attention over its input, so
cost grows ~quadratically with length (measured: 61s->0.2s, 255s->2.1s, but 698s->466s). Windowing keeps
each inference short (~1-2s) and stitches speaker ids across windows via the overlap region, so an
11-min call takes ~10s instead of ~8min — and no longer trips the client timeout.

  .venv-nemo/bin/python backend/sortformer_service.py
"""
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

os.environ.setdefault("TMPDIR", "/state/asr/data/nemo_tmp")  # .nemo + window temp wavs, not the 1 GB tmpfs
from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: E402

PORT = int(os.getenv("SORTFORMER_PORT", "5531"))
SR = 16000
WINDOW = float(os.getenv("SORTFORMER_WINDOW_SEC", "160"))   # per-inference window length (s)
OVERLAP = float(os.getenv("SORTFORMER_OVERLAP_SEC", "20"))  # window overlap, used to stitch speaker ids
STEP = WINDOW - OVERLAP

print("[sortformer] loading model...", flush=True)
_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1").to("cuda").eval()
_lock = threading.Lock()  # serialize GPU inference
print("[sortformer] ready on 127.0.0.1:%d (window=%ss overlap=%ss)" % (PORT, WINDOW, OVERLAP), flush=True)


def _load_16k_mono(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def _diarize_file(path):
    """Raw Sortformer on a whole file -> [{start,end,speaker}] (local times)."""
    raw = _model.diarize(audio=[path], batch_size=1)[0]
    segs = []
    for line in raw:
        p = line.split()
        if len(p) >= 3:
            segs.append({"start": float(p[0]), "end": float(p[1]), "speaker": p[2]})
    return segs


def _ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _stitch(local, prev_tail, next_gid, known):
    """Map a window's local speaker ids -> global ids. Greedy bipartite match on overlap with the
    previous window's global segments; a local speaker with no overlap is assigned to an UNUSED known
    global (e.g. the other party who was just silent at the boundary) before ever minting a new id —
    this stops one speaker being re-created as a spurious new id across windows ("Speaker 3")."""
    lspks = sorted({s["speaker"] for s in local})
    gspks = sorted({g["speaker"] for g in prev_tail})
    ov = {}
    for l in lspks:
        for g in gspks:
            tot = sum(_ov(s["start"], s["end"], gg["start"], gg["end"])
                      for s in local if s["speaker"] == l
                      for gg in prev_tail if gg["speaker"] == g)
            if tot > 0:
                ov[(l, g)] = tot
    mapping, used = {}, set()
    for _, l, g in sorted(((v, l, g) for (l, g), v in ov.items()), reverse=True):
        if l not in mapping and g not in used:
            mapping[l] = g
            used.add(g)
    free = [g for g in known if g not in used]   # existing globals not yet claimed this window
    for l in lspks:
        if l in mapping:
            continue
        if free:
            mapping[l] = free.pop(0)
        else:
            mapping[l] = "speaker_%d" % next_gid[0]
            next_gid[0] += 1
        used.add(mapping[l])
    return mapping


def diarize(audio_path):
    wav = _load_16k_mono(audio_path)
    dur = len(wav) / SR
    with _lock:
        if dur <= WINDOW + OVERLAP:
            return _diarize_file(audio_path)  # short: one shot (fast path, unchanged)

        out, prev_tail, next_gid, known = [], [], [0], set()
        w0 = 0.0
        while w0 < dur:
            w1 = min(w0 + WINDOW, dur)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", dir=os.environ["TMPDIR"], delete=False)
            tmp.close()
            try:
                sf.write(tmp.name, wav[int(w0 * SR):int(w1 * SR)], SR)
                local = _diarize_file(tmp.name)
            finally:
                os.unlink(tmp.name)
            for s in local:               # window-local -> global time
                s["start"] += w0
                s["end"] += w0
            if w0 == 0.0:                 # first window: assign fresh global ids in first-seen order
                mapping = {}
                for s in local:
                    mapping.setdefault(s["speaker"], "speaker_%d" % len(mapping))
                next_gid[0] = len(mapping)
            else:                          # later windows: map to prior ids via the overlap region
                mapping = _stitch(local, prev_tail, next_gid, known)
            known.update(mapping.values())
            remapped = [{"start": s["start"], "end": s["end"], "speaker": mapping[s["speaker"]]}
                        for s in local]
            emit_from = 0.0 if w0 == 0.0 else (w0 + OVERLAP)  # avoid duplicating prev window's region
            for s in remapped:
                st = max(s["start"], emit_from)
                if s["end"] > st:
                    out.append({"start": st, "end": s["end"], "speaker": s["speaker"]})
            nxt = w0 + STEP               # next window's overlap region = [nxt, nxt+OVERLAP]
            prev_tail = [s for s in remapped if s["end"] > nxt]
            w0 = nxt
        out.sort(key=lambda x: x["start"])
        return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send(200, {"status": "ok"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            path = req.get("audio_path")
            if not path or not os.path.isfile(path):
                return self._send(400, {"error": "audio_path missing or not found: %s" % path})
            self._send(200, {"segments": diarize(path)})
        except Exception as e:
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
