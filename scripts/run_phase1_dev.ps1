# Run Phase-1 development benchmark (proposed five models).
# Usage:
#   powershell -File scripts/run_phase1_dev.ps1
#   powershell -File scripts/run_phase1_dev.ps1 -Config configs/experiments/phase1_dev_large.yaml

param(
  [string]$Config = "configs/experiments/phase1_dev.yaml"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"

$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Missing .venv — create venv and install requirements first." }

& $py -m src.data.prepare
& $py -m src.training.train --config $Config
& $py -m src.evaluation.evaluate
& $py -m src.reporting.generate_phase1_report
& $py -m src.reporting.generate_selection_report

Write-Host "Done. See results/phase1/ and reports/."
Write-Host "Remember: development runs are not official until model list is confirmed."
