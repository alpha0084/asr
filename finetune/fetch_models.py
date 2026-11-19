"""Pre-fetch every eval/train model into the HF cache, reliably.

The box's connection to huggingface.co stalls mid-download (tiny config files hang
at 0%), which is what keeps killing the eval — not the models or the code. This
script snapshot-downloads each model with resume + retries + a hard per-request
timeout, so a stalled connection becomes a retriable error instead of an infinite
0% hang. Run it ONCE to warm the cache (mounted at /root/.cache/huggingface, i.e.
the persistent D:\\project\\ft-hfcache volume); then run the eval OFFLINE:

    python finetune/fetch_models.py                # all models
    python finetune/fetch_models.py whisper nllb   # subset (indic is usually cached already)
    HF_HUB_OFFLINE=1 python finetune/translate_eval.py --systems whisper-translate,indicconformer-nllb

Re-run on a stall — downloads resume where they left off; already-complete repos return instantly.
"""
import os, sys, time

# Convert a hung connection into a fast, retriable error instead of a forever-0% stall.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
# hf_transfer (when present) can swallow the timeout and hang — keep the plain, timeout-honoring path.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

MODELS = {
    "whisper":  "openai/whisper-large-v3",
    "nllb":     "facebook/nllb-200-distilled-600M",
    "seamless": "facebook/seamless-m4t-v2-large",
    "indic":    "ai4bharat/indic-conformer-600m-multilingual",
}


def fetch(repo):
    tok = os.environ.get("HF_TOKEN")
    for attempt in range(1, 9):
        try:
            p = snapshot_download(repo, token=tok, max_workers=1)  # resume is the default
            print(f"  OK  {repo}\n      -> {p}", flush=True)
            return True
        except Exception as ex:
            wait = min(30, 2 ** attempt)
            print(f"  retry {attempt}/8 {repo}: {type(ex).__name__}: {str(ex)[:100]}  (sleep {wait}s)", flush=True)
            time.sleep(wait)
    print(f"  FAILED {repo} after 8 attempts — re-run to resume", flush=True)
    return False


def main():
    want = sys.argv[1:] or list(MODELS)
    ok = True
    for key in want:
        repo = MODELS.get(key, key)
        print(f"\n== fetch {key} ({repo}) ==", flush=True)
        ok = fetch(repo) and ok
    print("\nALL DONE — now run the eval with HF_HUB_OFFLINE=1" if ok
          else "\nSOME FAILED — re-run this script to resume the stalled repos", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
