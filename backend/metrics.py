"""Live system metrics for the admin dashboard.

Cheap, dependency-free probes (no psutil): GPU via nvidia-smi, RAM via /proc/meminfo,
loaded LLMs via the Ollama /api/ps endpoint, API traffic via an in-process rolling
counter, and worker/queue load via the durable queue. Everything is best-effort —
any probe that fails returns None/empty so the dashboard degrades gracefully.
"""
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from collections import deque

from . import db, jobs
from .config import WORKER_CONCURRENCY

_HIT_WINDOW = 300.0                 # keep 5 min of request timestamps
_hits: deque = deque()
_hit_lock = threading.Lock()


def record_hit():
    """Called by the HTTP middleware for every API request (traffic-load signal)."""
    now = time.time()
    with _hit_lock:
        _hits.append(now)
        cutoff = now - _HIT_WINDOW
        while _hits and _hits[0] < cutoff:
            _hits.popleft()


def _traffic() -> dict:
    now = time.time()
    with _hit_lock:
        cutoff = now - _HIT_WINDOW
        while _hits and _hits[0] < cutoff:
            _hits.popleft()
        times = list(_hits)
    last_10 = sum(1 for t in times if t >= now - 10)
    last_60 = sum(1 for t in times if t >= now - 60)
    return {"rps": round(last_10 / 10.0, 2), "rpm": last_60, "last_5min": len(times)}


def _gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        gpus = []
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 5:
                mu, mt = int(float(p[2])), int(float(p[3]))
                gpus.append({"name": p[0], "util": int(float(p[1])),
                             "mem_used": mu, "mem_total": mt,
                             "mem_pct": round(100 * mu / mt) if mt else 0,
                             "temp": int(float(p[4]))})
        return gpus or None
    except Exception:
        return None


def _ram():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.strip().split()[0])  # kB
        total = info.get("MemTotal", 0) / 1048576
        avail = info.get("MemAvailable", 0) / 1048576
        used = max(0.0, total - avail)
        swap_t = info.get("SwapTotal", 0) / 1048576
        swap_f = info.get("SwapFree", 0) / 1048576
        return {"total_gb": round(total, 1), "used_gb": round(used, 1),
                "avail_gb": round(avail, 1), "used_pct": round(100 * used / total) if total else 0,
                "swap_used_gb": round(swap_t - swap_f, 1), "swap_total_gb": round(swap_t, 1)}
    except Exception:
        return None


def _ollama_loaded():
    """Models currently resident in the dedicated Ollama (VRAM), via /api/ps."""
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/ps", timeout=4) as r:
            data = json.load(r)
        return [{"name": m.get("name"),
                 "vram_mb": round((m.get("size_vram") or m.get("size") or 0) / 1048576)}
                for m in data.get("models", [])]
    except Exception:
        return []


def _workers():
    c = db.queue_counts()
    running = jobs.queue_status().get("running", [])
    active = len(running)
    return {"concurrency": WORKER_CONCURRENCY,
            "active": active,
            "load_pct": round(100 * active / WORKER_CONCURRENCY) if WORKER_CONCURRENCY else 0,
            "overloaded": (c.get("queued", 0) > 0 and active >= WORKER_CONCURRENCY),
            "queued": c.get("queued", 0), "running": c.get("running", 0),
            "done": c.get("done", 0), "error": c.get("error", 0),
            "active_stages": c.get("active_stages", 0),
            "jobs": running}


def snapshot() -> dict:
    return {
        "ts": int(time.time()),
        "gpu": _gpu(),
        "ram": _ram(),
        "models_loaded": _ollama_loaded(),
        "traffic": _traffic(),
        "workers": _workers(),
    }
