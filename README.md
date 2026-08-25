# Explainable AI — Diabetes Risk Prediction

Research-grade pipeline for **preliminary diabetes risk screening** with interpretable ML (SHAP + LIME), a FastAPI backend, and a path to a Flutter mobile client.

**Remote:** https://github.com/EshaanS013/Diabetes_XAI (private)

> **Not a diagnostic device.** Outputs are screening aids. Explanations are associative, not causal. SHAP/LIME agreement is explanation concordance, not predictive confidence.

## What this repo contains

- Leakage-aware data pipeline for CDC BRFSS2015 diabetes binary indicators
- Model registry for **10** algorithms; Phase-1 runs a configurable **five**
- Training / tuning / calibration / metrics / reporting CLIs
- SHAP + LIME + concordance utilities
- FastAPI prediction service (Dockerized)
- Research paper draft + panel outline with **TBD** slots for measured results only
- Tests for metrics, splits, leakage-sensitive preprocessing, and API health

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements/base.txt

python scripts/download_dataset.py
python -m src.data.prepare
pytest

# Development Phase-1 (proposed five; not official):
powershell -File scripts/run_phase1_dev.ps1
# Larger development subsample:
powershell -File scripts/run_phase1_dev.ps1 -Config configs/experiments/phase1_dev_large.yaml
```

### API

```bash
# After a trained artifact exists at artifacts/production/model.joblib
uvicorn api.app.main:app --reload
```

### Mobile

Flutter client source lives in `mobile/flutter_app/` (install Flutter SDK to run).

## Documentation

| Doc | Purpose |
|---|---|
| `docs/source_of_truth.md` | Document conflicts & Phase-1 decisions |
| `docs/experiment_protocol.md` | How experiments must be run |
| `docs/ethics.md` | Ethics / XAI limits for paper & panel |
| `docs/data_card.md` | Dataset card |
| `docs/implementation_plan.md` | Phased build plan |
| `research/paper/phase1_draft.md` | Paper draft (no fabricated metrics) |
| `presentation/phase1/outline.md` | September panel outline |

## Proposed Phase-1 five (pending confirmation)

1. Logistic Regression  
2. Random Forest  
3. XGBoost  
4. LightGBM  
5. MLP  

Confirm or edit `configs/phase1_models.yaml` before publishing official results.

## License

MIT — see `LICENSE`. Respect upstream dataset terms separately.
