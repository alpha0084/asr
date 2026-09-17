# ASR — Windows host hand-off

Deploy the **ASR tool** as its own isolated Docker stack on the GPU machine. This
runs it **separately from the production / whisper-ui container**, on its own network,
Postgres, Ollama, and persistent volumes — so recreating production never touches ASR,
and ASR data no longer gets wiped on container rebuilds.

**This does NOT touch production.** It creates brand-new containers/volumes on a
private network; it does not modify or restart the existing whisper-ui / GpuServer
containers, the production tunnel, or the dialer.

---

## 0. Prerequisites (one-time)
- **Docker Desktop** installed and running, with **GPU support** enabled
  (Settings → Resources → WSL/GPU; `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` should list the RTX 5090).
- **git** installed.
- The **secret values** for `.env` — provided to you separately by the ASR owner
  (HF token, API key, admin email/password, webhook secret).

## 1. Get the code
```powershell
cd D:\project
git clone -b development https://github.com/alpha0084/asr.git asr-standalone
cd asr-standalone\deploy
```
(Already cloned? `git pull` inside it.)

## 2. Create the .env
```powershell
copy .env.example .env
notepad .env
```
Fill in the values the owner gave you. Do **not** change `DATABASE_URL`, `OLLAMA_HOST`,
or `ASR_BACKEND` — the compose file sets those to the internal services.

## 3. Bring it up
```powershell
docker compose up -d --build
```
First build downloads torch/whisperx (~a few GB) — one time. Then pull the LLMs:
```powershell
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull llama3.2:3b
```

## 4. Verify
```powershell
docker compose ps
curl.exe -s -o NUL -w "app=%{http_code}`n" http://127.0.0.1:5530/docs   # expect 200
```
Open `http://127.0.0.1:5530/admin` → log in with the `ADMIN_EMAIL` / `ADMIN_PASSWORD`
from `.env`.

## 5. Make `asr.cnits.co` public — pick ONE

**Option A — reuse the existing prod tunnel (quickest).**
In the prod cloudflared config on `D:\project\whisper-ui\deploy\gpu-server1\cloudflared.yml`,
add this **above** the `- service: http_status:404` line, then restart that cloudflared:
```yaml
  - hostname: asr.cnits.co
    service: http://127.0.0.1:5530
```
(Shares the prod connector's fate — if it drops, ASR drops with it.)

**Option B — dedicated tunnel (fully independent, recommended).**
1. Cloudflare dashboard → Zero Trust → Networks → Tunnels → **Create tunnel**.
2. Add a public hostname: `asr.cnits.co` → `http://app:5530`.
3. Copy the tunnel **token** into `CLOUDFLARE_TUNNEL_TOKEN` in `.env`.
4. `docker compose --profile tunnel up -d`
Now ASR has its own tunnel — independent of production.

## 6. Keep it running 24/7
- Docker Desktop → Settings → General → **“Start Docker Desktop when you log in.”**
- Set the machine to **auto-login** (or run Docker as a service) so a reboot brings it back.
- The compose services already use `restart: unless-stopped`, so they auto-start with Docker.
- If there's a **nightly shutdown** task in Windows Task Scheduler, disable it (or use
  BIOS Wake-on-LAN) — otherwise the box (and production) go down every night.

## Everyday ops
```powershell
docker compose logs -f app          # app logs
docker compose restart app          # restart just the API
docker compose down                 # stop (DATA KEPT)
docker compose down -v              # stop AND DELETE data volumes — DO NOT unless wiping
docker compose exec postgres pg_dump -U asr_admin asr > asr-backup.sql   # backup DB
```

## Data lives in named volumes (survive recreation)
`asr_pgdata` (DB), `asr_ollama` (LLMs), `asr_appdata` (whisper cache + recordings).
`docker volume ls | findstr asr`. They persist across `up --build` / recreation. Only
`docker compose down -v` deletes them.

## After this is verified
The **temporary** ASR install on `/state` (in the GpuServer1 container) can be retired —
this standalone stack replaces it and is the permanent home.

## Contact
Questions on the app itself → the ASR owner. This stack is self-contained; you only
manage Docker on this host.
