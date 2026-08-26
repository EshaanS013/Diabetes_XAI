# Model Card — Logistic Regression (Phase-1 official candidate)

| Field | Value |
|---|---|
| Model name | Logistic Regression |
| Model version | phase1-official-lr-v1 |
| Algorithm | Logistic Regression (lbfgs, class_weight=balanced) |
| Dataset | CDC BRFSS2015 Diabetes Binary Health Indicators |
| Population | US adults (survey); not validated on Indian clinical data |
| Features | 21 |
| Target | Diabetes_binary |
| Split | 70/15/15 stratified, seed 42 |
| Training | Full training partition + in-train SMOTE; validation-tuned F1 threshold |
| Experiment | 20260825T171739Z_4aa740a6 |
| Random seed | 42 |
| Decision threshold | 0.59 |
| Artifact | rtifacts/production/model.joblib |

## Official held-out test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.783 |
| Precision | 0.353 |
| Recall | 0.670 |
| Specificity | 0.801 |
| F1 | 0.462 |
| ROC-AUC | 0.824 |
| PR-AUC | 0.422 |

## Intended use
Preliminary diabetes risk screening aid — **not a diagnostic tool**.

## Out of scope
Diagnosis, treatment decisions, emergency triage, India-population clinical claims without local validation.

See docs/ethics.md.
