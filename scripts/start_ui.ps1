<#
.SYNOPSIS
  Start the Agent Hub UI (Streamlit) in the background, then open it.

.DESCRIPTION
  Idempotent: if something already listens on the port, it just opens the browser.
  The server runs as a hidden, detached process, so it keeps running after this
  window, a terminal, or Claude Code closes. Logs: logs/streamlit.log.

.PARAMETER NoBrowser
  Start (or check) the server without opening a browser tab. Used at logon.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start_ui.ps1
#>
param(
    [switch]$NoBrowser,
    [int]$Port = 8502
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo '.venv\Scripts\streamlit.exe'
$logDir = Join-Path $repo 'logs'
$url = "http://localhost:$Port"

function Test-Up {
    try { return (Invoke-WebRequest "$url/_stcore/health" -UseBasicParsing -TimeoutSec 2).Content -eq 'ok' }
    catch { return $false }
}

if (-not (Test-Up)) {
    if (-not (Test-Path $exe)) { throw "Streamlit not found at $exe. Create the venv first (see README)." }
    New-Item -ItemType Directory -Force $logDir | Out-Null
    Start-Process -FilePath $exe -WorkingDirectory $repo -WindowStyle Hidden `
        -ArgumentList @('run', 'app/ui.py', '--server.headless', 'true', '--server.port', "$Port") `
        -RedirectStandardOutput (Join-Path $logDir 'streamlit.log') `
        -RedirectStandardError (Join-Path $logDir 'streamlit.err.log')
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Up)) {
        if ((Get-Date) -gt $deadline) { throw "UI did not come up within 60s. See $logDir\streamlit.err.log" }
        Start-Sleep -Milliseconds 500
    }
}

if (-not $NoBrowser) { Start-Process $url }
