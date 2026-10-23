# ASR git auto-deploy (single-shot; run by a Scheduled Task every ~2 min).
# Polls origin/<branch>; when the remote SHA moves, pulls + rebuilds + recreates the stack,
# then health-gates on /docs and rolls back to the last known-good image if it comes up unhealthy.
# In-flight jobs requeue (durable Postgres queue), so a rebuild mid-processing is safe.
# This is the ONLY way code reaches the box — never hand-edit on the box.
#
# One-time install (Administrator PowerShell; NOTE: no -RepetitionDuration — Win10/11 reject MaxValue):
#   $a = New-ScheduledTaskAction -Execute powershell.exe `
#        -Argument "-NoProfile -ExecutionPolicy Bypass -File D:\project\asr-standalone\deploy\autodeploy.ps1"
#   $t = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 2)
#   $p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
#   Register-ScheduledTask -TaskName asr-autodeploy -Action $a -Trigger $t -Principal $p
#
$ErrorActionPreference = "Stop"     # NOTE: does NOT catch native-exe failures — we check $LASTEXITCODE explicitly
$repo   = "D:\project\asr-standalone"
$branch = "development"
$log    = "D:\project\asr-autodeploy.log"           # outside the repo so it never dirties the tree
$lock   = "D:\project\asr-autodeploy.lock"
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -FilePath $log -Append -Encoding utf8 }

# single-instance guard: skip if a deploy is already running (a slow rebuild can outlast the 2-min tick)
if (Test-Path $lock) {
  if (((Get-Date) - (Get-Item $lock).LastWriteTime).TotalMinutes -lt 60) { exit 0 }  # in progress; skip
}
New-Item -ItemType File -Path $lock -Force | Out-Null
try {
  Set-Location $repo
  git fetch origin $branch --quiet
  if ($LASTEXITCODE -ne 0) { Log "git fetch failed ($LASTEXITCODE) - skipping"; exit 1 }
  $local  = (git rev-parse HEAD).Trim()
  $remote = (git rev-parse "origin/$branch").Trim()
  if ($local -eq $remote) { exit 0 }                 # nothing new

  Log "change $local -> $remote; deploying"
  git checkout $branch --quiet
  git pull --ff-only --quiet
  if ($LASTEXITCODE -ne 0) { Log "git pull failed ($LASTEXITCODE) - ABORT, old code still running"; exit 1 }

  Set-Location "$repo\deploy"
  docker compose build app 2>&1 | Out-File $log -Append -Encoding utf8
  if ($LASTEXITCODE -ne 0) { Log "docker build failed ($LASTEXITCODE) - ABORT, old image still running"; exit 1 }
  docker compose up -d 2>&1 | Out-File $log -Append -Encoding utf8
  if ($LASTEXITCODE -ne 0) { Log "docker up failed ($LASTEXITCODE) - ABORT"; exit 1 }

  # health gate — up to 2 min for the app to answer /docs
  $ok = $false
  for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 10
    try { if ((Invoke-WebRequest -UseBasicParsing http://localhost:5530/docs -TimeoutSec 10).StatusCode -eq 200) { $ok = $true; break } } catch {}
  }
  if ($ok) { Log "deploy OK at $remote"; exit 0 }

  # unhealthy -> roll back app to the last known-good image (retag -> :latest, recreate without building)
  Log "deploy UNHEALTHY at $remote - rolling back"
  $kg = (docker images --filter "reference=asr-app:known-good-*" --format "{{.Repository}}:{{.Tag}}" | Select-Object -First 1)
  if ($kg) {
    docker tag $kg asr-app:latest
    docker compose up -d --no-build app 2>&1 | Out-File $log -Append -Encoding utf8
    Log "rolled back app to $kg (HEAD stays at $remote; fix forward with a new commit)"
  } else {
    Log "NO known-good image to roll back to - MANUAL ATTENTION NEEDED"
  }
}
catch { Log "autodeploy ERROR: $_" }
finally { Remove-Item $lock -Force -ErrorAction SilentlyContinue }
