$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python "scripts\run_report_assets.py" @args
