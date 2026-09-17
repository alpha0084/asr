# Standalone ASR deployment (isolated from production)

Runs the ASR tool as its **own** Docker stack — separate container, network, Postgres,
Ollama, and **named volumes** — so it is decoupled from the production/whisper-ui
container. Recreating or rebuilding production no longer touches ASR, and ASR data
survives container recreation (the `/state` wipe problem is gone).

## What you get
| Service | Image | Purpose | Persistence |
|---|---|---|---|
| `app` | built from `deploy/Dockerfile` | FastAPI + whisperx/pyannote (GPU) | `appdata` volume (HF/whisper cache, recordings) |
| `postgres` | `postgres:18` | transcriptions, runs, admins, API keys + per-key webhooks | `pgdata` volume |
| `ollama` | `ollama/ollama` | diarization/summary/analytics LLMs (GPU) | `ollama` volume |
| `cloudflared` *(optional)* | `cloudflare/cloudflared` | dedicated tunnel for `asr.cnits.co` | — |

Everything is on a private `asr` network; only the app's `5530` is published (host
loopback). The GPU is shared with the host (Docker `gpus: all`).

## Deploy
```bash
cd deploy
cp .env.example .env          # fill in HF_TOKEN, API_KEY, ADMIN_EMAIL/PASSWORD, ...
docker compose up -d --build  # first build pulls torch/whisperx (~a few GB, one time)

# pull the LLMs into the ollama volume (one time)
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull llama3.2:3b

# check
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5530/docs   # 200
```
First login: `ADMIN_EMAIL` / `ADMIN_PASSWORD` from `.env` → then add more admins and
create per-environment API keys (each with its own webhook) in the admin UI.

## Exposing `asr.cnits.co` — two options
**A) Reuse the existing prod tunnel (simplest):** add one ingress rule to the prod
cloudflared config, pointing `asr.cnits.co` at this app:
```yaml
  - hostname: asr.cnits.co
    service: http://127.0.0.1:5530
```
(then restart that cloudflared). Shares the prod connector's fate, but no new tunnel.

**B) Dedicated tunnel (full isolation — recommended):** in the Cloudflare dashboard
create a tunnel, add a public hostname `asr.cnits.co → http://app:5530`, copy its token
into `CLOUDFLARE_TUNNEL_TOKEN` in `.env`, then:
```bash
docker compose --profile tunnel up -d
```
Now ASR has its **own** tunnel — independent of production entirely.

## Everyday ops
```bash
docker compose logs -f app          # app logs
docker compose restart app          # restart just the API
docker compose exec ollama ollama list
docker compose down                 # stop (volumes KEPT — data safe)
docker compose down -v              # stop AND delete volumes (wipes data — careful)
```

## Why this fixes the recurring wipe
The old deploy lived on `/state`, an **anonymous** Docker volume tied to the production
container — recreated empty whenever that container was rebuilt. Here, `pgdata` /
`ollama` / `appdata` are **named** volumes owned by *this* compose project, so they
persist across recreation and rebuilds. For belt-and-suspenders you can point them at
host bind mounts instead (e.g. `- D:\asr\pgdata:/var/lib/postgresql/data`).

## Notes
- **GPU:** shared with the host. For hard isolation use a second GPU and set
  `CUDA_VISIBLE_DEVICES` per service, or run this stack on a dedicated AI box.
- **RAM:** keep `WORKER_CONCURRENCY=2` unless the box has clear headroom.
- **Backups:** `docker compose exec postgres pg_dump -U asr_admin asr > asr.sql`.
