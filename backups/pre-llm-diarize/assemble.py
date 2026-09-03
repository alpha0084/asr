"""Stage 4 — merge transcription + diarization into speaker-labeled turns + talk stats."""
from collections import defaultdict


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(segments: list, dia_turns: list) -> list:
    """Tag each transcription segment with the speaker it overlaps most."""
    for seg in segments:
        best_spk, best_ov = None, 0.0
        for t in dia_turns:
            ov = _overlap(seg["start"], seg["end"], t["start"], t["end"])
            if ov > best_ov:
                best_ov, best_spk = ov, t["speaker"]
        seg["speaker"] = best_spk or "SPEAKER_?"
    return segments


def build_turns(segments: list) -> list:
    """Collapse consecutive same-speaker segments into readable turns."""
    turns = []
    for seg in segments:
        if not seg["text"]:
            continue
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turns[-1]["end"] = seg["end"]
            turns[-1]["text"] += " " + seg["text"]
        else:
            turns.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })
    return turns


def talk_stats(dia_turns: list, duration: float) -> dict:
    """Per-speaker talk time, share, and talk-to-listen ratio (from diarization)."""
    talk = defaultdict(float)
    for t in dia_turns:
        talk[t["speaker"]] += t["end"] - t["start"]
    total_talk = sum(talk.values()) or 1.0
    per_speaker = {
        spk: {
            "talk_seconds": round(secs, 1),
            "talk_share_pct": round(100 * secs / total_talk, 1),
        }
        for spk, secs in sorted(talk.items())
    }
    return {
        "duration_seconds": round(duration, 1),
        "speech_seconds": round(total_talk, 1),
        "silence_seconds": round(max(0.0, duration - total_talk), 1),
        "num_speakers": len(talk),
        "per_speaker": per_speaker,
    }
