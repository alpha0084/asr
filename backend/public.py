"""Public API response shaping — the clean JSON third parties receive."""


def _role_of(roles: dict, spk: str):
    if roles.get("agent") == spk:
        return "agent"
    if roles.get("customer") == spk:
        return "customer"
    return None


def public_result(job: dict) -> dict:
    """Reshape an internal job into the documented public payload."""
    out = {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
    }
    if job.get("status") == "error":
        out["error"] = job.get("error")
        return out

    r = job.get("result") or {}
    roles = r.get("roles", {}) or {}
    out.update({
        "audio_path": r.get("input"),
        "filename": r.get("filename"),
        "language": r.get("language"),
        "translated_to": r.get("target_language"),
        "duration_seconds": r.get("duration_seconds"),
        "processing_seconds": r.get("elapsed_seconds"),
        "roles": roles,
        "speakers": r.get("stats", {}).get("per_speaker", {}),
        "speaker_wise": [
            {
                "speaker": t["speaker"],
                "role": _role_of(roles, t["speaker"]),
                "start": round(t["start"], 2),
                "end": round(t["end"], 2),
                "text": t["text"],
                "text_translated": t.get("text_translated", ""),
            }
            for t in r.get("turns", [])
        ],
        "full_text": r.get("full_text", ""),
        "full_text_translated": r.get("full_text_translated", ""),
        "summary": r.get("summary", {}),
        "analytics": r.get("analytics", {}),
    })
    return out


def status_result(job: dict) -> dict:
    """Lightweight polling payload: progress + which stages are ready."""
    r = job.get("result") or {}
    return {
        "task_id": job.get("id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "detail": job.get("detail"),
        "summary_status": job.get("summary_status"),
        "analytics_status": job.get("analytics_status"),
        "error": job.get("error"),
        "language": r.get("language"),
        "duration_seconds": r.get("duration_seconds"),
        "transcript_ready": bool(r.get("turns")),
        "summary_ready": bool((r.get("summary") or {}).get("summary")),
        "analytics_ready": bool((r.get("analytics") or {}).get("scorecard")),
    }


def transcript_result(job: dict) -> dict:
    r = job.get("result") or {}
    roles = r.get("roles", {}) or {}
    return {
        "task_id": job.get("id"),
        "status": job.get("status"),
        "filename": r.get("filename"),
        "language": r.get("language"),
        "translated_to": r.get("target_language"),
        "duration_seconds": r.get("duration_seconds"),
        "roles": roles,
        "speakers": r.get("stats", {}).get("per_speaker", {}),
        "speaker_wise": [
            {
                "speaker": t["speaker"], "role": _role_of(roles, t["speaker"]),
                "start": round(t["start"], 2), "end": round(t["end"], 2),
                "text": t["text"], "text_translated": t.get("text_translated", ""),
            }
            for t in r.get("turns", [])
        ],
        "full_text": r.get("full_text", ""),
        "full_text_translated": r.get("full_text_translated", ""),
    }


def summary_result(job: dict) -> dict:
    r = job.get("result") or {}
    return {
        "task_id": job.get("id"),
        "summary_status": job.get("summary_status"),
        "summary": r.get("summary", {}),
    }


def analytics_result(job: dict) -> dict:
    r = job.get("result") or {}
    return {
        "task_id": job.get("id"),
        "analytics_status": job.get("analytics_status"),
        "analytics": r.get("analytics", {}),
    }


def task_list_item(job: dict) -> dict:
    return {
        "task_id": job.get("id"),
        "filename": job.get("filename"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "summary_status": job.get("summary_status"),
        "analytics_status": job.get("analytics_status"),
        "created_at": job.get("created_at"),
    }
