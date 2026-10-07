#!/bin/bash
# Keep the warm Sortformer diarization service up (isolated .venv-nemo). Idempotent: only starts it if
# 127.0.0.1:5531 isn't already answering. Mirrors start-asr.sh; run under supervisor like asr-keepalive.
if [ "$(id -u)" = 0 ]; then exec runuser -u asr -- "$0" "$@"; fi
cd /state/asr
export HOME=/state/asr-home HF_HOME=/state/asr/data/hf XDG_CACHE_HOME=/state/asr/data/cache TMPDIR=/state/asr/data/nemo_tmp
if ! curl -s -m3 http://127.0.0.1:5531/ >/dev/null 2>&1; then
  nohup .venv-nemo/bin/python backend/sortformer_service.py > data/sortformer.log 2>&1 &
  for i in $(seq 1 90); do sleep 1; curl -s -m3 http://127.0.0.1:5531/ >/dev/null 2>&1 && break; done
fi
echo "sortformer=$(curl -s -m3 -o /dev/null -w '%{http_code}' http://127.0.0.1:5531/)"
