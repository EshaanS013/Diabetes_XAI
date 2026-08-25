# Enhancing Clinical Decision Support through Explainable AI: Interpretable Machine Learning for Diabetes Risk Prediction

**Status:** Phase-1 draft — experimental results marked TBD until measured  
**Disclaimer:** Screening aid research; not a diagnostic system.

## Abstract

TBD - generated after experimental run. This paper will report a leakage-free comparison of five machine-learning algorithms for diabetes risk prediction on the CDC BRFSS2015 binary health-indicators dataset, coupled with SHAP and LIME explanations and an explicit ethical analysis of explanation limits.

## 1. Introduction

Diabetes imposes a large and growing burden globally, including in India. Predictive models can estimate risk from structured health indicators, but black-box outputs are difficult to trust in clinical and patient-facing settings. This project builds an explainable screening pipeline that returns both a calibrated risk estimate and dual local explanations (SHAP and LIME), delivered through a mobile-oriented architecture.

## 2. Related Work

Recent 2025 diabetes-XAI studies pair classifiers with SHAP/LIME (see proposal literature survey). Gaps motivating this work include mobile delivery, latency-aware explanation budgets, and using SHAP–LIME agreement as an **explanation concordance** signal rather than a correctness claim.

## 3. Problem Statement

Build a reproducible ML + XAI framework for preliminary diabetes risk screening using self-reportable features, with transparent methodology, ethical constraints, and a path to mobile deployment.

## 4. Dataset

CDC BRFSS2015 Diabetes Binary Health Indicators (~253,680 rows; 21 features). Chosen over PIMA because features are enterable without laboratory tests and the sample is larger and covers both sexes.

**Dataset statistics:** TBD - generated after experimental run (see `reports/data_quality_report.md` after `python -m src.data.prepare`).

## 5. Methodology

### 5.1 Split and leakage controls

Stratified 70/15/15 train/validation/test. Preprocessing fitted on train only. SMOTE applied only within training / CV training folds. Test reserved for final evaluation.

### 5.2 Models (Phase 1)

Five algorithms (exact list pending confirmation; proposed set in `configs/phase1_models.yaml`):

1. Logistic Regression  
2. Random Forest  
3. XGBoost  
4. LightGBM  
5. Multilayer Perceptron  

The codebase also registers Naive Bayes, k-NN, SVM, Gradient Boosting, and CatBoost for later phases.

### 5.3 Metrics

Accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier score, ECE, confusion matrices. Primary tuning objective: F1 (imbalance-aware). Model selection will not use accuracy alone.

### 5.4 Explainability

SHAP and LIME local attributions; top-k concordance (Jaccard / exact set match). Concordance is **not** predictive confidence.

## 6. Results

> **Label:** Phase-1 **development** benchmark (25k stratified train subsample).  
> **Not** the official full-data Phase-1 result. Replace after confirmation + full run.

Measured development comparison (`results/phase1/model_comparison.csv`):

| Model | Recall | F1 | ROC-AUC | Precision | Threshold |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.656 | 0.465 | 0.824 | 0.360 | 0.60 |
| Random Forest | 0.643 | 0.449 | 0.815 | 0.345 | 0.38 |
| XGBoost | 0.624 | 0.454 | 0.818 | 0.356 | 0.34 |
| LightGBM | 0.658 | 0.448 | 0.817 | 0.340 | 0.32 |
| MLP | 0.627 | 0.456 | 0.819 | 0.358 | 0.58 |

On this development run, Logistic Regression led F1 and ROC-AUC; LightGBM led recall.  
**Production choice remains TBD** pending full-data official run and multi-criteria review (see `reports/model_selection_report.md`).

### 6.1 Model comparison table

See above (development only).

### 6.2 Calibration

Calibration fitted on validation; classification metrics above use uncalibrated scores with validation-tuned thresholds. Calibrated Brier/ECE are stored per-model under experiment artifacts.

### 6.3 Explanation concordance

Measured on **15** held-out test instances for the promoted **development** Logistic Regression artifact (`results/phase1/explainability/shap_lime_local.json`):

| Aggregate metric | Value |
|---|---:|
| Mean Jaccard (top-3) | 0.66 |
| Exact top-3 set agreement rate | 0.47 |

Example instance top-3 (both methods): GenHlth, BMI, HighBP.

Interpretation (mandatory): concordance is **explanation-method agreement only**, not predictive confidence or clinical truth. SHAP/LIME are not causal.

## 7. Ethical Issues and Concerns

### 7.1 False reassurance and false alarms

False negatives may discourage care-seeking; false positives may induce anxiety. Screening ethics requires explicit discussion of these errors.

### 7.2 Explainability concerns

Explanations can be unstable (especially LIME), disagree across methods, and be misread as causal or as proof of correctness.

### 7.3 Limitations of SHAP and LIME

SHAP cost and correlation sensitivity; LIME stochasticity and local fidelity limits; neither substitutes for clinical validation.

### 7.4 Risks of misleading explanations

Users may over-trust polished natural-language summaries. Templates are deterministic and marked `clinical_review_status: pending` until reviewed.

### 7.5 Clinical-use limitations

Not a diagnosis. US survey training data may not transfer to Indian patients without local validation. Probability displays require calibration checks. Human clinicians remain responsible for decisions.

## 8. Discussion

TBD - generated after experimental run (tradeoffs among recall, precision, calibration, latency, explainability).

## 9. Conclusion

This Phase-1 draft establishes a reproducible, leakage-aware experimental framework and an ethical stance for explainable diabetes risk screening. Measured results will replace all TBD sections after the confirmed five-model experiment.

## References

See proposal Section 25 and `docs/reference/Explainable_AI_Proposal.pdf`. Key anchors: IDF Diabetes Atlas (2025); BRFSS/CDC diabetes health indicators; 2025 SHAP/LIME diabetes studies cited in the proposal; Grinsztajn et al. (2022) on tabular deep learning.
