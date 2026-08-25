# Model Card — Logistic Regression (Phase-1 development candidate)

| Field | Value |
|---|---|
| Model name | Logistic Regression |
| Model version | `phase1-dev-lr-v1` |
| Algorithm | Logistic Regression (`lbfgs`, `class_weight=balanced`) |
| Dataset | CDC BRFSS2015 Diabetes Binary Health Indicators |
| Population | US adults (survey self-report); **not** validated on Indian clinical data |
| Features | 21 (see `configs/data_schema.yaml`) |
| Target | `Diabetes_binary` (0 = no diabetes; 1 = prediabetes or diabetes) |
| Split | 70/15/15 stratified, seed 42 (`split-v1`) |
| Training regime | **Development**: stratified 25k train subsample + SMOTE in-train; validation-tuned F1 threshold |
| Training date | 2026-08-24 (experiment `20260824T183931Z_3485fc59`) |
| Random seed | 42 |
| Decision threshold | 0.60 (tuned on validation for F1; default 0.5 retained for calibration diagnostics) |
| Artifact | `artifacts/production/model.joblib` |

## Measured development test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.789 |
| Precision | 0.360 |
| Recall | 0.656 |
| Specificity | 0.811 |
| F1 | 0.465 |
| ROC-AUC | 0.824 |
| PR-AUC | 0.420 |

> These are **development** numbers, not official full-data Phase-1 publication metrics.

## Calibration

Isotonic calibration fitted on validation only. User-facing % risk should prefer calibrated probabilities after full-data validation of Brier/ECE.

## Explainability

- Offline: SHAP + LIME (`src.explainability.generate`)
- Online API: fast ablation + optional LIME
- Concordance (dev, n=15): mean Jaccard top-3 ≈ 0.66; exact top-3 match rate ≈ 0.47
- Concordance ≠ predictive confidence; attributions ≠ causation
- Text templates: `clinical_review_status: pending`

## Intended use

Preliminary **diabetes risk screening aid** for research / decision-support prototyping.

## Out of scope

- Diagnosis
- Treatment decisions
- Emergency triage
- Claims of India-population clinical validity without local validation

## Known limitations

- US BRFSS survey population and self-report bias
- Class imbalance; threshold choice strongly affects FN/FP tradeoff
- Development train subsample — retrain on full data before freezing
- Explanation methods can disagree and mislead

## Ethical considerations

See `docs/ethics.md`. False negatives can cause false reassurance. Never present as a diagnostic device.

## Deployment

- FastAPI: `uvicorn api.app.main:app`
- Env: `MODEL_ARTIFACT_PATH=artifacts/production/model.joblib`

**Not a diagnostic tool.**
