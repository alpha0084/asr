#!/bin/bash
# Wait for Postgres, then launch the API. Schema + first-admin are created on startup
# by the app itself (db.init_db + admin_auth.bootstrap).
set -e

echo "[entrypoint] waiting for Postgres at ${DATABASE_URL}"
python3 - <<'PY'
import os, time, psycopg
url = os.environ["DATABASE_URL"]
for i in range(90):
    try:
        psycopg.connect(url, connect_timeout=3).close()
        print("[entrypoint] Postgres ready"); break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("[entrypoint] Postgres not reachable after 90s")
PY

exec python3 -m uvicorn backend.server:app --host 0.0.0.0 --port 5530
