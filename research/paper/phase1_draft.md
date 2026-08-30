# Enhancing Clinical Decision Support through Explainable AI: Interpretable Machine Learning for Diabetes Risk Prediction

**Status:** Phase-1 draft — official full-data benchmark complete  
**Disclaimer:** Screening aid research; not a diagnostic system.

## Abstract

We present a leakage-aware comparison of five machine-learning algorithms for preliminary diabetes risk screening on the CDC BRFSS2015 binary health-indicators dataset (253,680 respondents; 21 self-reportable features). Models were tuned with stratified cross-validation optimising F1 under class imbalance, with decision thresholds selected on a held-out validation split and final metrics reported on a sacred test split. Logistic Regression achieved the highest F1 (0.462) and ROC-AUC (0.824) among the five candidates, with competitive recall (0.670). We pair the selected model with SHAP and LIME local explanations and report explanation concordance separately from predictive performance. An ethical analysis addresses false reassurance, explanation instability, non-causality, and limits of transferring US survey data to Indian clinical contexts. **This system is a screening aid, not a diagnostic device.**

## 1. Introduction

Diabetes imposes a large and growing burden globally, including in India. Predictive models can estimate risk from structured health indicators, but black-box outputs are difficult to trust in clinical and patient-facing settings. This project builds an explainable screening pipeline that returns both a calibrated risk estimate and dual local explanations (SHAP and LIME), delivered through a mobile-oriented architecture.

## 2. Related Work

Recent diabetes-XAI studies pair classifiers with SHAP/LIME (see proposal literature survey). Gaps motivating this work include mobile delivery, latency-aware explanation budgets, and using SHAP–LIME agreement as an **explanation concordance** signal rather than a correctness claim.

## 3. Problem Statement

Build a reproducible ML + XAI framework for preliminary diabetes risk screening using self-reportable features, with transparent methodology, ethical constraints, and a path to mobile deployment.

## 4. Dataset

CDC BRFSS2015 Diabetes Binary Health Indicators (253,680 rows; 21 features). Chosen over PIMA because features are enterable without laboratory tests and the sample is larger and covers both sexes.

**Dataset statistics** (from `reports/data_quality_report.md`):

| Statistic | Value |
|---|---:|
| Rows | 253,680 |
| Features | 21 |
| Class 0 (no diabetes) | 218,334 (86.1%) |
| Class 1 (diabetes/prediabetes) | 35,346 (13.9%) |
| Missing values | 0 |
| SHA-256 hash | `4b702ecd2e80e431148cf29f12158928a2b401e97a65ddb7293ad5e6dc1a6f60` |

## 5. Methodology

### 5.1 Split and leakage controls

Stratified 70/15/15 train/validation/test (177,575 / 38,052 / 38,053). Preprocessing fitted on train only. SMOTE applied only within training / CV training folds. Test reserved for final evaluation.

### 5.2 Models (Phase 1)

Confirmed five algorithms (`configs/phase1_models.yaml`):

1. Logistic Regression  
2. Random Forest  
3. XGBoost  
4. LightGBM  
5. Multilayer Perceptron  

The codebase also registers Naive Bayes, k-NN, SVM, Gradient Boosting, and CatBoost for later phases.

### 5.3 Metrics

Accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier score, ECE, confusion matrices. Primary tuning objective: F1 (imbalance-aware). Model selection does not use accuracy alone.

### 5.4 Explainability

SHAP and LIME local attributions; top-k concordance (Jaccard / exact set match). Concordance is **not** predictive confidence.

## 6. Results

> **Label:** Official full-data Phase-1 benchmark (confirmed five models; full training partition; validation-tuned F1 thresholds).  
> Experiment: `20260830T084820Z_cf03e581` (local reproducible rerun).

Measured held-out **test** comparison (`results/phase1/model_comparison.csv`):

| Model | Accuracy | Recall | F1 | ROC-AUC | Precision | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.783 | **0.670** | **0.462** | **0.824** | 0.353 | 0.801 |
| MLP | 0.787 | 0.643 | 0.457 | 0.820 | 0.354 | 0.810 |
| LightGBM | 0.783 | 0.651 | 0.455 | 0.818 | 0.350 | 0.804 |
| Random Forest | 0.793 | 0.619 | 0.455 | 0.814 | 0.360 | 0.822 |
| XGBoost | 0.781 | 0.647 | 0.452 | 0.808 | 0.347 | 0.803 |

**Selected production candidate (Phase 1):** Logistic Regression — highest F1 and ROC-AUC on the official test split, competitive recall, lowest inference latency (~0.32 ms/row), and native coefficient interpretability plus SHAP/LIME layer. LightGBM remains competitive on recall. Random Forest leads on accuracy and specificity but underperforms on recall/F1 for screening use. Final clinical deployment would still require local validation and calibration review.

Figures: combined ROC curves, PR curves, metric bars, and per-model confusion matrices in `results/phase1/figures/`.

### 6.1 Confusion matrices (test split, validation-tuned thresholds)

| Model | TN | FP | FN | TP | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 26,239 | 6,512 | 1,750 | 3,552 | 0.670 | 0.801 |
| LightGBM | 26,339 | 6,412 | 1,850 | 3,452 | 0.651 | 0.804 |
| MLP | 26,540 | 6,211 | 1,893 | 3,409 | 0.643 | 0.810 |
| XGBoost | 26,438 | 6,313 | 1,899 | 3,403 | 0.642 | 0.807 |
| Random Forest | 26,913 | 5,838 | 2,020 | 3,282 | 0.619 | 0.822 |

*(Values derived from `results/phase1/test_metrics.json`; sensitivity = TP/(TP+FN), specificity = TN/(TN+FP).)*

### 6.2 Calibration

Isotonic calibration fitted on validation only. Classification metrics above use uncalibrated scores with validation-tuned thresholds. Per-model Brier/ECE are in `model_comparison.csv`. Logistic Regression shows higher ECE (0.237) than tree/boosting models; probability displays require calibration review before clinical-facing use.

### 6.3 Explanation concordance

Measured on **30** held-out test instances for the promoted Logistic Regression artifact (`results/phase1/explainability/shap_lime_local.json`):

| Aggregate metric | Value |
|---|---:|
| Mean Jaccard (top-3) | 0.42 |
| Exact top-3 set agreement rate | 0.03 |

Example instance: SHAP top-3 GenHlth, BMI, HighBP; LIME overlapped on GenHlth/BMI (Jaccard 0.5).

Interpretation (mandatory): concordance is **explanation-method agreement only**, not predictive confidence or clinical truth. SHAP/LIME are not causal. The modest exact-match rate underscores why dual explainers are useful as a disagreement flag, not a certainty score.

## 7. Ethical Issues and Concerns

### 7.1 False reassurance and false alarms

False negatives may discourage care-seeking; false positives may induce anxiety. At the selected LR threshold, ~1,750 false negatives occur on the test split — a material screening risk. Ethics requires explicit discussion of these errors and clear UI disclaimers.

### 7.2 Explainability concerns

Explanations can be unstable (especially LIME), disagree across methods, and be misread as causal or as proof of correctness. Our measured exact top-3 agreement rate of ~3% demonstrates that explanations must not be presented as unified clinical truth.

### 7.3 Limitations of SHAP and LIME

SHAP cost and correlation sensitivity; LIME stochasticity and local fidelity limits; neither substitutes for clinical validation. Kernel/permutation SHAP on pipelines adds latency unsuitable for unbounded mobile use without budgets.

### 7.4 Risks of misleading explanations

Users may over-trust polished natural-language summaries. Templates are deterministic and marked `clinical_review_status: pending` until reviewed.

### 7.5 Clinical-use limitations

Not a diagnosis. US survey training data may not transfer to Indian patients without local validation. Probability displays require calibration checks. Human clinicians remain responsible for decisions.

## 8. Discussion

All five models achieve similar accuracy (~0.78–0.79), but screening-relevant metrics diverge. Logistic Regression balances the best F1 and ROC-AUC with the highest recall among top performers, while Random Forest trades recall for specificity. Boosting models (XGBoost, LightGBM) sit mid-pack on F1/AUC; MLP is competitive but slowest to train (~34 min vs ~20 s for LR). No model eliminates the fundamental ~14% prevalence imbalance challenge: precision remains ~0.35 at useful recall levels.

For Phase 1 we prioritise **recall-aware F1 tuning** over raw accuracy, consistent with supervisor guidance and screening ethics. ROC-AUC alone does not measure clinical utility; threshold choice materially affects FN/FP tradeoffs. AWS student credits remain optional for later API deployment; Phase-1 training completed on local CPU.

Limitations: US BRFSS self-report data; no Indian validation cohort; explanation concordance measured on 30 instances only; calibration imperfect for LR.

## 9. Conclusion

Phase 1 delivers a reproducible, leakage-aware benchmark of five algorithms for explainable diabetes risk screening. Logistic Regression is the provisional Phase-1 candidate based on measured F1, ROC-AUC, recall, and latency. SHAP/LIME provide complementary local attributions with explicitly bounded interpretation. Future work: Flutter mobile client, FastAPI deployment (AWS optional), remaining five algorithms, and local clinical validation in India.

## References

See proposal Section 25 and `docs/reference/Explainable_AI_Proposal.pdf`. Key anchors: IDF Diabetes Atlas (2025); BRFSS/CDC diabetes health indicators; 2025 SHAP/LIME diabetes studies cited in the proposal; Grinsztajn et al. (2022) on tabular deep learning.
