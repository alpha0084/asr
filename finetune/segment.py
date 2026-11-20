"""Turn call-level (audio, English) pairs into <=30s clip-level training examples.

Whisper trains on <=30s clips, each with the English for THAT clip. Your data is full
calls (minutes) with one call-level English reference. This bridges the two:

  1. Run Whisper (translate, sequential long-form) on the full call -> English segments
     WITH timestamps (this is the only reliable source of audio<->time alignment).
  2. Distribute each segment's time across its words -> a (word, time) track of the HYP.
  3. Align your clean human reference English to that hyp track (English<->English, via
     difflib) so every reference word inherits a timestamp. This is the key trick: we
     never train on Whisper's noisy text, only borrow its *timing*; the TEXT is your
     ground-truth reference.
  4. Group reference words into <=--max-seconds windows by their timestamps, cut the audio
     for each window with ffmpeg, and emit clips + a clip-level dataset.json.

    python finetune/segment.py --data finetune/data/dataset.json --out finetune/data/clips
    python finetune/train.py --data finetune/data/clips/dataset.json --base openai/whisper-large-v3 --lora

APPROXIMATE by nature (timing comes from ASR segmentation + text alignment, not true
forced alignment). It is good enough to make call-level data trainable, and it only gets
better with volume. For the cleanest possible clips, prefer capturing data as short
(per-turn / <=30s) audio + English directly (e.g. from the QC feedback-correction loop).
"""
import argparse, json, re, subprocess, difflib, os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # same broken-xet workaround as the eval


def whisper_segments(wav):
    """English segments with (start, end) timestamps via sequential long-form decoding."""
    import torch
    from transformers import pipeline
    global _P
    try:
        _P
    except NameError:
        dev = 0 if torch.cuda.is_available() else -1
        dt = torch.float16 if torch.cuda.is_available() else torch.float32
        _P = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                      device=dev, torch_dtype=dt)
    r = _P(str(wav), return_timestamps=True, generate_kwargs={
        "task": "translate", "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6, "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)})
    out = []
    for c in r.get("chunks", []):
        ts = c.get("timestamp") or (None, None)
        if ts[0] is None:
            continue
        out.append((c["text"].strip(), float(ts[0]), float(ts[1] if ts[1] is not None else ts[0])))
    return out


def hyp_word_times(segments):
    """Spread each segment's [start,end] evenly across its words -> [(word, time), ...]."""
    track = []
    for text, s, e in segments:
        words = text.split()
        if not words:
            continue
        step = (e - s) / len(words)
        for i, w in enumerate(words):
            track.append((w, s + step * (i + 0.5)))
    return track


def assign_times(ref_words, hyp_track):
    """Give every reference word a timestamp by aligning ref<->hyp text (difflib), then
    interpolating monotonically across unmatched spans."""
    hyp_words = [w for w, _ in hyp_track]
    hyp_t = [t for _, t in hyp_track]
    norm = lambda xs: [re.sub(r"[^a-z0-9]", "", w.lower()) for w in xs]
    sm = difflib.SequenceMatcher(None, norm(ref_words), norm(hyp_words), autojunk=False)
    times = [None] * len(ref_words)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            times[a + k] = hyp_t[b + k]
    # fill leading/trailing/gap Nones by interpolation, keep non-decreasing
    known = [(i, t) for i, t in enumerate(times) if t is not None]
    if not known:
        return [i * 0.0 for i in range(len(ref_words))]
    for i in range(len(times)):
        if times[i] is None:
            prev = max([k for k in known if k[0] <= i], default=known[0], key=lambda x: x[0])
            nxt = min([k for k in known if k[0] >= i], default=known[-1], key=lambda x: x[0])
            if prev[0] == nxt[0]:
                times[i] = prev[1]
            else:
                frac = (i - prev[0]) / (nxt[0] - prev[0])
                times[i] = prev[1] + frac * (nxt[1] - prev[1])
    run = 0.0
    for i in range(len(times)):
        run = times[i] = max(run, times[i])
    return times


def cut(src, t0, t1, dst):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", str(src), "-ss", f"{t0:.2f}",
                    "-to", f"{t1:.2f}", "-ar", "16000", "-ac", "1", str(dst)],
                   capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="call-level dataset.json from build_dataset.py")
    ap.add_argument("--out", default="finetune/data/clips")
    ap.add_argument("--max-seconds", type=float, default=28.0)
    ap.add_argument("--min-words", type=int, default=4, help="drop windows with fewer ref words")
    ap.add_argument("--max-words", type=int, default=90, help="hard cap on ref words per clip "
                    "(keeps English under Whisper's 448-token decoder limit even if timing is off)")
    args = ap.parse_args()

    out = Path(args.out); (out / "wav").mkdir(parents=True, exist_ok=True)
    calls = [e for e in json.loads(Path(args.data).read_text()) if e.get("audio") and e.get("ref_en")]
    clips = []
    for e in calls:
        segs = whisper_segments(e["audio"])
        track = hyp_word_times(segs)
        ref_words = e["ref_en"].split()
        if not track or not ref_words:
            print(f"  {e['id']:12} skipped (no timing)"); continue
        times = assign_times(ref_words, track)
        # Fallback: if alignment collapsed (few anchors -> times span <half the call), spread
        # ref words uniformly across the call duration so windows cover the whole call instead
        # of cramming most of the text into a couple of windows (the sreethu_2 "4 clips" case).
        dur = e.get("duration_s") or 0
        span = times[-1] - times[0]
        if dur and span < 0.5 * dur:
            print(f"  {e['id']:12} alignment weak (span {span:.0f}s / {dur:.0f}s) -> uniform spacing")
            times = [dur * i / len(ref_words) for i in range(len(ref_words))]

        win_start_i, win_t0 = 0, times[0]
        made = 0
        def flush(end_i, t_end):
            nonlocal made
            words = ref_words[win_start_i:end_i]
            if len(words) < args.min_words:
                return
            t0 = max(0.0, times[win_start_i] - 0.15)
            t1 = min(e.get("duration_s", t_end) or t_end, t_end + 0.15)
            if t1 - t0 < 0.4:
                return
            cid = f"{e['id']}_{made:03d}"
            wav = out / "wav" / f"{cid}.wav"
            cut(e["audio"], t0, t1, wav)
            clips.append({"id": cid, "audio": str(wav), "ref_en": " ".join(words),
                          "duration_s": round(t1 - t0, 2), "parent": e["id"]})
            made += 1

        for i in range(1, len(ref_words)):
            if times[i] - win_t0 >= args.max_seconds or (i - win_start_i) >= args.max_words:
                flush(i, times[i - 1])
                win_start_i, win_t0 = i, times[i]
        flush(len(ref_words), times[-1])
        print(f"  {e['id']:12} {made} clips  ({len(segs)} whisper segs)")

    (out / "dataset.json").write_text(json.dumps(clips, indent=2))
    secs = sum(c["duration_s"] for c in clips)
    print(f"\n{len(clips)} clips -> {out/'dataset.json'}  ({secs/60:.1f} min total)")
    print("train:  python finetune/train.py --data %s --base openai/whisper-large-v3 --lora" % (out / "dataset.json"))


if __name__ == "__main__":
    main()
