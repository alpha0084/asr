"""Shared orchestration.

Split into stages so the web UI can run them on demand:
  transcribe_only()  -> speaker-wise transcript + full text (+ translation). No LLM QA.
  add_summary()      -> attach the conversation summary to an existing result.
  add_analytics()    -> attach the QA scorecard to an existing result.
  process()          -> all-in-one (used by the public API / CLI).
"""
import json
import time

from ..config import DATA_DIR, WHISPER_MODEL
from ..utils import fmt_ts, speaker_label
from . import analyze, assemble, diarize, ingest, preprocess, summarize, transcribe, translate

# Stages for the first (transcribe) pass; summary/analytics run separately.
STAGES = ["Ingesting", "Preprocessing", "Transcribing", "Diarizing", "Assembling", "Translating"]


def transcribe_only(src, model=None, language=None, speakers=None,
                    target_language=None, on_stage=None, on_progress=None) -> dict:
    """Ingest → preprocess → transcribe → diarize → assemble → translate.

    Returns speaker-wise turns + full text (+ translation). No summary/analytics.
    """
    model = model or WHISPER_MODEL
    target_language = (target_language or "").strip() or None

    def stage(name):
        if on_stage:
            on_stage(name)

    def progress(detail):
        if on_progress:
            on_progress(detail)

    t0 = time.time()

    stage("Ingesting")
    audio_path = ingest.resolve_input(src)

    stage("Preprocessing")
    meta = preprocess.probe(audio_path)
    work_dir = DATA_DIR / audio_path.stem
    wav = preprocess.to_wav(audio_path, work_dir / "audio16k.wav")
    waveform, sr = preprocess.load_waveform(wav)

    stage("Transcribing")
    tr = transcribe.transcribe(
        waveform, sr, model_name=model, language=language,
        on_progress=lambda frac, cur, tot: progress(f"{fmt_ts(cur)} / {fmt_ts(tot)} ({int(frac*100)}%)"),
    )

    stage("Diarizing")
    dia = diarize.diarize(waveform, sr, num_speakers=speakers)

    stage("Assembling")
    segs = assemble.assign_speakers(tr["segments"], dia)
    raw_turns = assemble.build_turns(segs)
    stats = assemble.talk_stats(dia, meta["duration"])

    label_map: dict = {}
    turns = [
        {**t, "speaker_raw": t["speaker"], "speaker": speaker_label(t["speaker"], label_map)}
        for t in raw_turns
    ]
    stats["per_speaker"] = {
        speaker_label(k, label_map): v for k, v in stats["per_speaker"].items()
    }

    if target_language and translate.same_language(tr["language"], target_language):
        target_language = None      # already in target language — skip
    if target_language:
        stage("Translating")
        try:
            turns = translate.translate_turns(
                turns, target_language,
                on_progress=lambda i, n: progress(f"batch {i}/{n}"),
            )
        except Exception as e:
            target_language = f"{target_language} (failed: {type(e).__name__})"

    result = {
        "input": str(audio_path),
        "filename": audio_path.name,
        "language": tr["language"],
        "language_probability": tr["language_probability"],
        "target_language": target_language,
        "duration_seconds": meta["duration"],
        "stats": stats,
        "roles": {},
        "turns": turns,
        "full_text": " ".join(t["text"] for t in turns),
        "full_text_translated":
            " ".join(t.get("text_translated", "") for t in turns) if target_language else "",
        "summary": {},
        "analytics": {},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    _save(work_dir, result)
    return result


def add_summary(result: dict, on_stage=None) -> dict:
    """Run the conversation summary on an existing transcript result (in place)."""
    if on_stage:
        on_stage("Summarizing")
    try:
        summary = summarize.summarize(result["turns"])
    except Exception as e:
        summary = {"error": f"{type(e).__name__}: {e}"}
    result["summary"] = summary
    if summary.get("roles"):
        result["roles"] = summary["roles"]
    _resave(result)
    return result


def add_analytics(result: dict, on_stage=None, on_progress=None) -> dict:
    """Run the QA scorecard on an existing transcript result (in place)."""
    if on_stage:
        on_stage("Analyzing")
    # Reuse roles from a prior summary if present; otherwise infer them cheaply.
    roles = result.get("roles") or (result.get("summary") or {}).get("roles")
    if not roles:
        try:
            roles = summarize.infer_roles(result["turns"])
            result["roles"] = roles
        except Exception:
            roles = {}
    try:
        qa = analyze.analyze(
            result["turns"], roles=roles,
            on_progress=(lambda i, n, cat: on_progress(f"category {i}/{n}: {cat}")) if on_progress else None,
        )
    except Exception as e:
        qa = {"error": f"{type(e).__name__}: {e}"}
    result["analytics"] = qa
    _resave(result)
    return result


def process(src, model=None, language=None, speakers=None, target_language=None,
            analytics=True, on_stage=None, on_progress=None) -> dict:
    """All-in-one: transcript + (optionally) summary + QA. Used by the public API/CLI."""
    result = transcribe_only(src, model=model, language=language, speakers=speakers,
                             target_language=target_language, on_stage=on_stage, on_progress=on_progress)
    if analytics:
        add_summary(result, on_stage=on_stage)
        add_analytics(result, on_stage=on_stage, on_progress=on_progress)
    return result


def _save(work_dir, result):
    out_json = work_dir / "transcript.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["out_json"] = str(out_json)


def _resave(result):
    if result.get("out_json"):
        from pathlib import Path
        Path(result["out_json"]).write_text(json.dumps(
            {k: v for k, v in result.items() if k != "out_json"}, indent=2, ensure_ascii=False))
