"""Warm Sortformer diarization microservice.

Loads nvidia/diar_sortformer_4spk-v1 ONCE and serves diarization over a local HTTP socket, so the prod
pipeline (a different venv, no NeMo) can use it via `pipeline/sortformer_client.py`. Run with the
ISOLATED NeMo python (.venv-nemo) — NeMo never enters the prod venv. Binds 127.0.0.1:5531.

  .venv-nemo/bin/python backend/sortformer_service.py
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("TMPDIR", "/state/asr/data/nemo_tmp")  # .nemo extracts here, not the 1 GB tmpfs
from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: E402

PORT = int(os.getenv("SORTFORMER_PORT", "5531"))
print("[sortformer] loading model...", flush=True)
_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1").to("cuda").eval()
_lock = threading.Lock()  # serialize GPU inference
print("[sortformer] ready on 127.0.0.1:%d" % PORT, flush=True)


def _diarize(audio_path: str) -> list:
    with _lock:
        raw = _model.diarize(audio=[audio_path], batch_size=1)[0]
    segs = []
    for line in raw:
        p = line.split()
        if len(p) >= 3:
            segs.append({"start": float(p[0]), "end": float(p[1]), "speaker": p[2]})
    return segs


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
            self._send(200, {"segments": _diarize(path)})
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
