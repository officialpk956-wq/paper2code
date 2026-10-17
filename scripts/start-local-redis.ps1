# Start Docker Desktop (working around its stale-socket crash) and bring up Redis.
# Docker 4.81 on Windows leaves Docker\run\dockerInference behind on shutdown and then
# crashes on the next launch ("initializing Inference manager ... file cannot be accessed").
# Moving the run dir aside is non-destructive; Docker recreates it. See
# docs/LOCAL_UPLOAD_RECOVERY_2026-10-16.md.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start-local-redis.ps1

$run = Join-Path $env:LOCALAPPDATA 'Docker\run'
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $run) { Rename-Item $run "run.backup-$(Get-Date -Format yyyyMMdd-HHmmss)" }
    Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    for ($i = 0; $i -lt 36; $i++) {
        Start-Sleep 5
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { break }
    }
    if ($LASTEXITCODE -ne 0) { throw 'Docker engine did not start within 3 minutes.' }
}
Set-Location (Split-Path $PSScriptRoot)
docker compose up -d redis
docker compose exec -T redis redis-cli ping
