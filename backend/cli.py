"""Phase 1 CLI — one recording -> speaker-wise transcript.

Usage:
    python -m backend.cli <file-or-URL> [--model small] [--language hi] [--speakers 2]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .config import WHISPER_MODEL
from .pipeline import runner
from .utils import fmt_ts


def print_report(r: dict):
    print("\n" + "=" * 70)
    print(f"TRANSCRIPT  ·  {Path(r['input']).name}")
    print(f"language: {r['language']}  ·  duration: {fmt_ts(r['duration_seconds'])}"
          f"  ·  processed in {fmt_ts(r['elapsed_seconds'])}")
    s = r["stats"]
    print(f"speakers: {s['num_speakers']}  ·  speech: {fmt_ts(s['speech_seconds'])}"
          f"  ·  silence: {fmt_ts(s['silence_seconds'])}")
    for spk, v in s["per_speaker"].items():
        print(f"    {spk}: {fmt_ts(v['talk_seconds'])} ({v['talk_share_pct']}%)")
    print("-" * 70)
    for t in r["turns"]:
        print(f"[{fmt_ts(t['start'])}–{fmt_ts(t['end'])}] {t['speaker']}: {t['text']}")
        if t.get("text_translated"):
            print(f"          ↳ {t['text_translated']}")

    summary = r.get("summary") or {}
    if summary and "error" not in summary:
        print("\n" + "-" * 70 + "\nSUMMARY")
        print(f"  {summary.get('summary','')}")
        print(f"  outcome: {summary.get('outcome','?')}  ·  "
              f"sentiment: {summary.get('customer_sentiment',{}).get('overall','?')}")
        for ai in summary.get("action_items", []):
            print(f"  • action: {ai}")

    qa = r.get("analytics") or {}
    if qa and "error" not in qa:
        print("\n" + "-" * 70 + f"\nQA SCORE: {qa.get('overall_score','?')}/100"
              f"  (compliance_fail={qa.get('compliance_fail')})")
        for row in qa.get("scorecard", []):
            print(f"  [{row.get('score')}] {row.get('checkpoint')}: {row.get('verdict')}"
                  f" — {(row.get('evidence') or '')[:60]}")
    print("=" * 70)
    print(f"\n✓ saved: {r['out_json']}")


def main():
    ap = argparse.ArgumentParser(description="ASR Phase 1 — speaker-wise transcript")
    ap.add_argument("input", help="audio/video file path or http(s)/presigned-S3 URL")
    ap.add_argument("--model", default=WHISPER_MODEL,
                    help="whisper model: tiny|base|small|medium|large-v3 (default: %(default)s)")
    ap.add_argument("--language", default=None,
                    help="source language code (e.g. hi, pa, en). Default: auto-detect")
    ap.add_argument("--speakers", type=int, default=None,
                    help="pin number of speakers (e.g. 2 for a 1:1 call)")
    ap.add_argument("--translate", default=None, metavar="LANG",
                    help="translate transcript into this language (e.g. English, Hindi)")
    ap.add_argument("--no-analytics", action="store_true",
                    help="skip summary + QA analytics (transcript only)")
    args = ap.parse_args()

    try:
        result = runner.process(
            args.input, model=args.model, language=args.language, speakers=args.speakers,
            target_language=args.translate, analytics=not args.no_analytics,
            on_stage=lambda name: print(f"▶ {name}", flush=True),
        )
    except Exception as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    print_report(result)


if __name__ == "__main__":
    main()
