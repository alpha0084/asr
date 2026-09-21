"""Cheap voice-activity pre-check — skip the GPU pipeline on blank / no-speech recordings.

Uses the Silero VAD that ships with faster-whisper (CPU, milliseconds, no model download), so a
dead-air / voicemail / hold-music recording never pays for Whisper + pyannote + the LLM. Runs on
the already-cleaned 16 kHz mono waveform, AFTER the loudness/denoise pass, so quiet-but-real
speech isn't mistaken for silence.
"""
from faster_whisper.vad import VadOptions, get_speech_timestamps


def speech_stats(waveform, sr) -> tuple[float, int]:
    """(total detected speech seconds, number of speech segments) in the waveform.

    Fail-OPEN: if the VAD itself errors we report 'infinite' speech so the gate never drops a
    real call — the worst case is we transcribe a blank one, never that we skip a good one.
    """
    try:
        ts = get_speech_timestamps(waveform, VadOptions(), sampling_rate=sr) or []
    except Exception:
        return (float("inf"), 1)
    total = sum((t["end"] - t["start"]) for t in ts) / float(sr or 16000)
    return (total, len(ts))
