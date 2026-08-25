# Experiment Protocol (Phase 1)

## Objective

Compare five machine-learning algorithms for binary diabetes-risk prediction on the CDC BRFSS2015 cleaned binary extract, under a leakage-free protocol, and produce measured metrics for the research paper and panel.

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

## Optimisation metric

- Primary tuning objective: **F1** (imbalanced screening context)
- Always report: accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier, ECE, confusion matrix

## Model selection policy

Do **not** pick the production model by accuracy alone. Discuss recall (false-negative burden), F1, ROC-AUC, calibration, latency, and explainability compatibility.

## Official vs development runs

- Official Phase-1 publication requires `selection_status: confirmed` in `configs/phase1_models.yaml`
- Development runs may proceed while pending confirmation; label outputs as non-official

## Reproducibility artifacts

Each experiment writes `artifacts/experiments/<id>/metadata.json` including seed, package versions, git commit (if available), and config snapshots.
