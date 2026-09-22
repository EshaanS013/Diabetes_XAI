# Experiment Protocol (Phase 1 + Phase 2)

## Objective

Compare the full registered ten-model suite for binary diabetes-risk prediction on the CDC BRFSS2015 cleaned binary extract and produce measured, reproducible metrics.

## Dataset

- Source: CDC BRFSS 2015 Diabetes Health Indicators (binary)
- Path: `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`
- Target: `Diabetes_binary` (0/1)
- Features: 21 self-reportable indicators (see `configs/data_schema.yaml`)

## Split

- 70% train / 15% validation / 15% test
- Stratified on target
- Seed: 42 (see `configs/base.yaml`)
- Indices saved under `artifacts/splits/`

## Leakage rules

- Fit scalers, imputers, BMI IQR caps **only** on training data
- SMOTE **only** inside training / CV training folds
- Hyperparameter selection uses CV on train; validation for calibration / optional threshold discussion
- **Test set untouched** until model configuration is frozen
- Rank candidates by training-fold CV F1. Test metrics are final retrospective evaluation and are not a model-selection objective.

### Exact-duplicate limitation

The original `split-v1` is row-random and contains exact feature-plus-target duplicates across partitions. Phase 2 retains it only so results remain directly comparable with the completed Phase 1. `results/phase2/duplicate_overlap_audit.json` quantifies the overlap. A separately versioned duplicate-grouped split and sensitivity rerun are required before clinical-performance claims.

## Optimisation metric

- Primary tuning objective: **F1** (imbalanced screening context)
- Always report: accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier, ECE, confusion matrix

## Model selection policy

Do **not** pick the production model from held-out test rankings or accuracy alone. The automated recommendation is based on CV F1; human review must also cover recall (false-negative burden), calibration, subgroup behavior, latency, and explainability compatibility.

## Official vs development runs

- Official Phase-1 publication requires `selection_status: confirmed` in `configs/phase1_models.yaml`
- Official Phase-2 publication requires `selection_status: confirmed` in `configs/phase2_models.yaml`
- Development runs may proceed while pending confirmation; label outputs as non-official

## Reproducibility artifacts

Each experiment writes `artifacts/experiments/<id>/metadata.json` including seed, package versions, git commit (if available), and config snapshots.
