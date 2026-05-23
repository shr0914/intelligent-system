$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python "scripts\evaluate_low_light_extension.py" @args
