# Run the official Phase-2 benchmark: the five registered models not run in Phase 1.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"

$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Missing .venv — create it and install requirements/base.txt first." }

& $py -m src.data.prepare
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py -m src.data.audit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py -m src.training.train --config configs/phase2_models.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py -m src.evaluation.evaluate --results-dir results/phase2 --title "Phase-2 remaining-five benchmark"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py scripts/aggregate_all_models.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Official Phase-2 complete. See results/phase2/ and results/all_models/."
