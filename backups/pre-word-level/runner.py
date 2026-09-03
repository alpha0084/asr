"""Shared orchestration — one recording -> speaker-wise transcript.

Used by both the CLI and the web server. `on_stage(label)` is an optional
callback so callers can surface progress (print, or update a job record).
"""
import json
import time

from ..config import DATA_DIR, WHISPER_MODEL
from ..utils import fmt_ts, speaker_label
from . import analyze, assemble, diarize, ingest, preprocess, summarize, transcribe, translate

STAGES = ["Ingesting", "Preprocessing", "Transcribing", "Diarizing", "Assembling",
          "Translating", "Summarizing", "Analyzing"]


def process(src, model=None, language=None, speakers=None,
            target_language=None, analytics=True, on_stage=None, on_progress=None) -> dict:
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

    # --- Phase 2: translation + summary + QA analytics (best-effort) ---
    if target_language and translate.same_language(tr["language"], target_language):
        # Call is already in the requested language — skip pointless self-translation.
        target_language = None
    if target_language:
        stage("Translating")
        try:
            turns = translate.translate_turns(
                turns, target_language,
                on_progress=lambda i, n: progress(f"batch {i}/{n}"),
            )
        except Exception as e:
            target_language = f"{target_language} (failed: {type(e).__name__})"

    summary, qa = {}, {}
    if analytics:
        stage("Summarizing")
        try:
            summary = summarize.summarize(turns)
        except Exception as e:
            summary = {"error": f"{type(e).__name__}: {e}"}
        stage("Analyzing")
        try:
            qa = analyze.analyze(
                turns, roles=summary.get("roles"),
                on_progress=lambda i, n, cat: progress(f"category {i}/{n}: {cat}"),
            )
        except Exception as e:
            qa = {"error": f"{type(e).__name__}: {e}"}

    result = {
        "input": str(audio_path),
        "filename": audio_path.name,
        "language": tr["language"],
        "language_probability": tr["language_probability"],
        "target_language": target_language,
        "duration_seconds": meta["duration"],
        "stats": stats,
        "roles": summary.get("roles", {}),
        "turns": turns,
        "full_text": " ".join(t["text"] for t in turns),
        "full_text_translated": " ".join(t.get("text_translated", "") for t in turns) if target_language else "",
        "summary": summary,
        "analytics": qa,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_json = work_dir / "transcript.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["out_json"] = str(out_json)
    return result
