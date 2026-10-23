# ASR git auto-deploy (single-shot; run by a Scheduled Task every ~2 min).
# Polls origin/<branch>; when the remote SHA moves, pulls + rebuilds + recreates the stack,
# then health-gates on /docs. On a healthy deploy it re-tags the image as the rollback target;
# on an unhealthy one it rolls back to the last healthy image. In-flight jobs requeue (durable
# Postgres queue), so a rebuild mid-processing is safe. This is the ONLY way code reaches the box.
#
# One-time seed (so a rollback has a target before the first healthy deploy):
#   docker tag <current-verified-image-id> asr-app:known-good
#
# One-time install (Administrator PowerShell; NOTE: no -RepetitionDuration — Win10/11 reject MaxValue):
#   $a = New-ScheduledTaskAction -Execute powershell.exe `
#        -Argument "-NoProfile -ExecutionPolicy Bypass -File D:\project\asr-standalone\deploy\autodeploy.ps1"
#   $t = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 2)
#   $p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
#   Register-ScheduledTask -TaskName asr-autodeploy -Action $a -Trigger $t -Principal $p
#
# NOTE: EAP is 'Continue', not 'Stop'. In PS 5.1, a native command's stderr under `2>&1`/`*>>`
# raises a terminating error when EAP='Stop' (docker compose writes progress to stderr at exit 0),
# which would make every deploy "fail". We gate on $LASTEXITCODE instead; cmdlets that must abort
# carry an explicit -ErrorAction Stop and are caught by the outer try/catch.
$ErrorActionPreference = "Continue"
$repo   = "D:\project\asr-standalone"
$branch = "development"
$good   = "asr-app:known-good"                      # rolling tag = last image that passed the health gate
$log    = "D:\project\asr-autodeploy.log"           # outside the repo so it never dirties the tree
$lock   = "D:\project\asr-autodeploy.lock"
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -FilePath $log -Append -Encoding utf8 }

# single-instance guard: skip if a deploy is already running (a slow rebuild can outlast the 2-min tick)
if (Test-Path $lock) {
  if (((Get-Date) - (Get-Item $lock).LastWriteTime).TotalMinutes -lt 60) { exit 0 }  # in progress; skip
}
New-Item -ItemType File -Path $lock -Force | Out-Null
try {
  Set-Location $repo -ErrorAction Stop
  git fetch origin $branch --quiet
  if ($LASTEXITCODE -ne 0) { Log "git fetch failed ($LASTEXITCODE) - skipping"; exit 1 }
  $local  = (git rev-parse HEAD).Trim()
  $remote = (git rev-parse "origin/$branch").Trim()
  if ($local -eq $remote) { exit 0 }                 # nothing new

  Log "change $local -> $remote; deploying"
  git checkout $branch --quiet
  git pull --ff-only --quiet
  if ($LASTEXITCODE -ne 0) { Log "git pull failed ($LASTEXITCODE) - ABORT, old code still running"; exit 1 }

  Set-Location "$repo\deploy" -ErrorAction Stop
  docker compose build app *>> $log                  # *>> = all streams to log; EAP=Continue so stderr won't throw
  if ($LASTEXITCODE -ne 0) { Log "docker build failed ($LASTEXITCODE) - ABORT, old image still running"; exit 1 }
  docker compose up -d *>> $log
  if ($LASTEXITCODE -ne 0) { Log "docker up failed ($LASTEXITCODE) - ABORT"; exit 1 }

  # health gate — up to 2 min for the app to answer /docs
  $ok = $false
  for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 10
    try { if ((Invoke-WebRequest -UseBasicParsing http://localhost:5530/docs -TimeoutSec 10).StatusCode -eq 200) { $ok = $true; break } } catch {}
  }
  if ($ok) {
    docker tag asr-app:latest $good *>> $log          # this image is now the known-good rollback target
    Log "deploy OK at $remote (tagged $good)"
    exit 0
  }

  Log "deploy UNHEALTHY at $remote - rolling back to $good"
  if (docker images -q $good) {
    docker tag $good asr-app:latest *>> $log
    docker compose up -d --no-build app *>> $log
    Log "rolled back app to $good (HEAD stays at $remote; fix forward with a new commit)"
  } else {
    Log "NO $good image to roll back to - MANUAL ATTENTION NEEDED"
  }
}
catch { Log "autodeploy ERROR: $_" }
finally { Remove-Item $lock -Force -ErrorAction SilentlyContinue }
