# ASR git auto-deploy (single-shot; run by a Scheduled Task every ~2 min).
# Polls origin/<branch>; when the remote SHA moves, pulls + rebuilds + recreates the stack,
# then health-gates on /docs. In-flight jobs requeue (durable Postgres queue), so a rebuild
# mid-processing is safe. This is the ONLY way code reaches the box — never hand-edit on the box.
#
# One-time install (PowerShell as the Docker Desktop user):
#   $a = New-ScheduledTaskAction -Execute powershell.exe `
#        -Argument "-NoProfile -ExecutionPolicy Bypass -File D:\project\asr-standalone\deploy\autodeploy.ps1"
#   $t = New-ScheduledTaskTrigger -Once -At (Get-Date) `
#        -RepetitionInterval (New-TimeSpan -Minutes 2) -RepetitionDuration ([TimeSpan]::MaxValue)
#   $p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
#   Register-ScheduledTask -TaskName asr-autodeploy -Action $a -Trigger $t -Principal $p
#
$ErrorActionPreference = "Stop"
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
  $local  = (git rev-parse HEAD).Trim()
  $remote = (git rev-parse "origin/$branch").Trim()
  if ($local -eq $remote) { exit 0 }                 # nothing new

  Log "change $local -> $remote; deploying"
  git checkout $branch --quiet
  git pull --ff-only --quiet                         # ff-only: fails safe if the box ever has local commits
  Set-Location "$repo\deploy"
  docker compose build app | Out-File $log -Append -Encoding utf8
  docker compose up -d     | Out-File $log -Append -Encoding utf8

  # health gate — up to 2 min for the app to answer /docs
  $ok = $false
  for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 10
    try { if ((Invoke-WebRequest -UseBasicParsing http://localhost:5530/docs -TimeoutSec 10).StatusCode -eq 200) { $ok = $true; break } } catch {}
  }
  if ($ok) { Log "deploy OK at $remote" }
  else     { Log "deploy UNHEALTHY at $remote - check 'docker compose logs app'; rollback image asr-app:known-good-*" }
}
catch { Log "autodeploy ERROR: $_" }
finally { Remove-Item $lock -Force -ErrorAction SilentlyContinue }
