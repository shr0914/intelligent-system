$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$Mode = "one-step"
$RemainingArgs = @()

if ($args.Count -gt 0) {
    if ($args[0] -eq "two-step" -or $args[0] -eq "--two-step") {
        $Mode = "two-step"
        if ($args.Count -gt 1) {
            $RemainingArgs = $args[1..($args.Count - 1)]
        }
    }
    elseif ($args[0] -eq "one-step" -or $args[0] -eq "--one-step") {
        if ($args.Count -gt 1) {
            $RemainingArgs = $args[1..($args.Count - 1)]
        }
    }
    else {
        $RemainingArgs = $args
    }
}

& $Python "scripts\run_gui.py" --mode $Mode --source 0 @RemainingArgs
