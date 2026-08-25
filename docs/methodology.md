# Methodology Overview

See also: `docs/experiment_protocol.md`, `docs/ethics.md`, `configs/base.yaml`.

## Pipeline

User/self-report features → validated schema → train-fitted preprocessing → model → optional validation-fitted calibration → risk probability → SHAP + LIME → concordance → deterministic safe-text templates → API/mobile.

## Anti-leakage

Split first. Fit transforms on train. SMOTE inside train/CV train folds only. Test is sacred.

## Models

Central registry in `src/models/registry.py` supports ten algorithms. Phase-1 subset is config-driven.

## Evaluation

Imbalance-aware primary tuning metric (F1) with full clinical-relevant metric suite and calibration diagnostics.

## Explainability

SHAP (global/local) + LIME (local) + Jaccard/top-k concordance. Concordance ≠ certainty. Attributions ≠ causation.
