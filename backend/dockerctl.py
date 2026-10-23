"""Docker control for the admin panel — read container stats + restart/stop/start the
asr stack's own containers, over the mounted /var/run/docker.sock.

Scoped to the compose project (label com.docker.compose.project=asr) so the panel can
never touch unrelated containers. Best-effort: if the socket is missing/unreachable/not
permitted (dev box, no mount, wrong DOCKER_GID), calls degrade gracefully — `available()`
returns False and `list_containers()` returns [] instead of raising.
"""
import logging
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

PROJECT = "asr"                                  # compose project (docker-compose.yml `name:`)
_PROJECT_LABEL = "com.docker.compose.project"
_SERVICE_LABEL = "com.docker.compose.service"

try:
    import docker
except Exception:                                # docker-py not installed (local dev)
    docker = None


def _client():
    if docker is None:
        raise RuntimeError("docker SDK not installed")
    return docker.from_env()


def available() -> bool:
    """True if the Docker daemon is reachable (socket mounted + permitted)."""
    try:
        _client().ping()
        return True
    except Exception as e:
        log.debug("docker unavailable: %s", e)
        return False


def _ours(client):
    return client.containers.list(all=True, filters={"label": f"{_PROJECT_LABEL}={PROJECT}"})


def _find(client, service: str):
    for ct in _ours(client):
        if ct.labels.get(_SERVICE_LABEL) == service or ct.name == service:
            return ct
    raise KeyError(f"no asr container for service {service!r}")


def _cpu_pct(stats: dict):
    """docker-stats CPU% from one sample (it carries precpu too)."""
    try:
        c, p = stats["cpu_stats"], stats["precpu_stats"]
        cpu_delta = c["cpu_usage"]["total_usage"] - p["cpu_usage"]["total_usage"]
        sys_delta = c.get("system_cpu_usage", 0) - p.get("system_cpu_usage", 0)
        ncpu = c.get("online_cpus") or len(c["cpu_usage"].get("percpu_usage") or []) or 1
        if sys_delta > 0 and cpu_delta >= 0:
            return round(cpu_delta / sys_delta * ncpu * 100, 1)
    except Exception:
        pass
    return None


def _mem(stats: dict):
    try:
        m = stats["memory_stats"]
        usage = m["usage"] - m.get("stats", {}).get("inactive_file", 0)
        limit = m.get("limit") or 0
        return round(usage / 1e6, 1), (round(usage / limit * 100, 1) if limit else None)
    except Exception:
        return None, None


def _row(ct, with_stats: bool) -> dict:
    state = ct.attrs.get("State", {})
    row = {
        "name": ct.name,
        "service": ct.labels.get(_SERVICE_LABEL, ""),
        "status": ct.status,                                   # running/exited/restarting/...
        "health": (state.get("Health") or {}).get("Status"),  # healthy/unhealthy/starting/None
        "image": (ct.image.tags[0] if ct.image.tags else ct.image.short_id),
        "restarts": state.get("RestartCount", 0),
        "started_at": state.get("StartedAt"),
        "cpu_pct": None, "mem_mb": None, "mem_pct": None,
    }
    if with_stats and ct.status == "running":
        try:
            s = ct.stats(stream=False)
            row["cpu_pct"] = _cpu_pct(s)
            row["mem_mb"], row["mem_pct"] = _mem(s)
        except Exception as e:
            log.debug("stats failed for %s: %s", ct.name, e)
    return row


def list_containers(with_stats: bool = True) -> list[dict]:
    """One row per asr-stack container: identity, status/health, restarts, image, live CPU/mem.
    Stats are gathered in parallel (each `stats` read blocks ~1s on the daemon)."""
    try:
        cts = _ours(_client())
        if with_stats:
            with ThreadPoolExecutor(max_workers=8) as ex:
                rows = list(ex.map(lambda ct: _row(ct, True), cts))
        else:
            rows = [_row(ct, False) for ct in cts]
        rows.sort(key=lambda r: r["service"])
        return rows
    except Exception as e:
        log.warning("list_containers failed: %s", e)
        return []


def logs(service: str, tail: int = 200) -> str:
    ct = _find(_client(), service)
    return ct.logs(tail=tail, timestamps=True).decode("utf-8", "replace")


def action(service: str, act: str) -> dict:
    if act not in ("restart", "stop", "start"):
        raise ValueError(f"unknown action {act!r}")
    # Guard the obvious foot-gun: stopping `app` kills the very container serving this panel
    # (restart:unless-stopped won't bring a manual stop back). Restart is fine.
    if service == "app" and act == "stop":
        raise ValueError("refusing to stop the app container (would kill the admin panel) — use restart")
    ct = _find(_client(), service)
    getattr(ct, act)()
    return {"ok": True, "service": service, "action": act}
