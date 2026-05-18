$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python "scripts\run_gui.py" --mode one-step --source 0 @args
