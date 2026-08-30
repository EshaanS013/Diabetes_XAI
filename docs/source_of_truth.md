# Source of Truth & Reconciliation

**Project:** Enhancing Clinical Decision Support through Explainable AI: Interpretable Machine Learning for Diabetes Risk Prediction

**Repository audit date:** 2026-08-24  
**Initial repository state:** empty workspace (no prior code). Reference PDFs ingested into `docs/reference/`.

## Documents consulted

1. `docs/reference/Explainable_AI_Proposal.pdf` (proposal)
2. `docs/reference/Master_Cursor_Prompt.pdf` (implementation master prompt)
3. Latest supervisor instructions summarized in the master prompt (Phase 1: five algorithms; paper + panel by early September)

## Priority order

When documents disagree, apply this order:

1. Latest supervisor instructions (Phase 1 milestone)
2. Master implementation prompt (engineering / scientific methodology)
3. Proposal document (research framing, stack, original 3-model design)

## Known inconsistencies

| Topic | Proposal | Master prompt / supervisor | Resolution in this repo |
|---|---|---|---|
| Number of models (Phase 1) | 3 (LR, RF, XGBoost) | 5 algorithms now; architecture for 10 | Architecture supports **10**; Phase 1 config lists **5 proposed** models |
| Which five? | N/A | **Not specified** | See `configs/phase1_models.yaml` — **proposed, pending confirmation** |
| Primary production model | XGBoost assumed | Must be decided experimentally | No hardcoded winner |
| Stack | Flutter + FastAPI + Firebase | Same, AWS optional for compute | Preserved; FastAPI scaffolded; Flutter placeholder |
| Results in paper | Expected metrics narrative | **Never fabricate results** | Paper uses `TBD - generated after experimental run` until measured |

## Phase-1 five — CONFIRMED (2026-08-25)

1. `logistic_regression` — linear interpretable baseline (proposal)
2. `random_forest` — bagging benchmark (proposal)
3. `xgboost` — boosting / proposal primary candidate
4. `lightgbm` — second booster for family comparison
5. `mlp` — non-tree neural baseline

Confirmed by project owner. Config: `configs/phase1_models.yaml` (`selection_status: confirmed`).

## Immediate September milestone

1. Train/compare five algorithms with full metrics (accuracy, precision, recall, F1, ROC, ROC-AUC, specificity, PR-AUC, Brier, ECE)
2. Draft research paper including ethics, SHAP/LIME limitations, clinical-use limits
3. Panel presentation materials
4. Do **not** invent metrics

## Blocking decisions

- [x] Exact five Phase-1 algorithms confirmed by supervisor/team
- [x] Dataset file placed under `data/raw/` (download script or manual Kaggle/UCI)
- [x] AWS account usage (optional) vs local training for Phase 1 — **local training used**
- [ ] Flutter kickoff timing (after model selection preferred)

## Non-negotiables

- Screening aid ≠ diagnosis
- Anti-leakage splits; SMOTE train-only
- No fabricated research results
- SHAP/LIME agreement ≠ predictive confidence
- Explanations ≠ causation
