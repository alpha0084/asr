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
