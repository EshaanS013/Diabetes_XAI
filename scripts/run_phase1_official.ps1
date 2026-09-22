# Run OFFICIAL Phase-1 benchmark (confirmed five models, full data).
# Usage:
#   powershell -File scripts/run_phase1_official.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"

$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Missing .venv — create venv and install requirements first." }

& $py -m src.data.prepare
& $py -m src.training.train --config configs/phase1_models.yaml
& $py -m src.evaluation.evaluate

$exp = Get-ChildItem artifacts/experiments -Directory | Where-Object { $_.Name -match '^\d{8}T' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$lrArtifact = Join-Path $exp.FullName "models/logistic_regression/artifact.joblib"
if (Test-Path $lrArtifact) {
  & $py -m src.explainability.generate --artifact $lrArtifact --strategy linear --n-samples 30
  & $py scripts/promote_model.py --artifact $lrArtifact
}

& $py -m src.reporting.generate_phase1_report
& $py -m src.reporting.generate_selection_report

Write-Host "Official Phase-1 complete. See results/phase1/ and research/paper/phase1_draft.md"
