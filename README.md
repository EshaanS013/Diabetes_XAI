# Explainable AI — Diabetes Risk Prediction

Research-grade pipeline for **preliminary diabetes risk screening** with interpretable ML (SHAP + LIME), a FastAPI backend, and a path to a Flutter mobile client.

**Remote (public):** https://github.com/EshaanS013/Diabetes_XAI

> **Not a diagnostic device.** Outputs are screening aids. Explanations are associative, not causal. SHAP/LIME agreement is explanation concordance, not predictive confidence.

## Ten-model benchmark complete

Phase 1 evaluated Logistic Regression, Random Forest, XGBoost, LightGBM, and MLP.
Phase 2 evaluates Gaussian Naive Bayes, KNN, linear SVM, histogram gradient boosting,
and CatBoost. Measured outputs are tracked in `results/phase1/`, `results/phase2/`,
and the CV-ranked combined table in `results/all_models/`.

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
pip install -r requirements/base.txt -c requirements/constraints-tested.txt

python scripts/download_dataset.py
python -m src.data.prepare
pytest

# Development smoke run:
powershell -File scripts/run_phase1_dev.ps1

# Official full-data benchmarks:
powershell -File scripts/run_phase1_official.ps1
powershell -File scripts/run_phase2_official.ps1
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

## Benchmark status

Both confirmed five-model phases have completed full-data runs on the frozen split.
The combined ranking uses training-fold CV F1; held-out test metrics are reported only
for final evaluation. No production model is auto-promoted from test performance.

Important limitation: `split-v1` contains exact duplicate rows across partitions.
See `results/phase2/duplicate_overlap_audit.json`. A duplicate-grouped sensitivity
benchmark, subgroup/fairness review, and clinical validation remain necessary before
deployment or clinical claims.

Paper draft: `research/paper/phase1_draft.md`  
Panel slides: `presentation/phase1/slides_draft.md`

## License

MIT — see `LICENSE`. Respect upstream dataset terms separately.
