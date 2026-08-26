# Phase 1 Panel — Slide Content Draft

**Status:** Official full-data Phase-1 metrics (confirmed five models).  
Share repo: https://github.com/EshaanS013/Diabetes_XAI

---

## 1. Title
**Enhancing Clinical Decision Support through Explainable AI**  
Interpretable ML for Diabetes Risk Prediction  
*Screening aid — not a diagnosis*

## 2. Motivation
- India’s diabetes burden is large; many cases undiagnosed
- Black-box scores lack actionable “why”
- Need mobile-ready, self-reportable inputs (no lab glucose)

## 3. Research question
Leakage-free explainable **risk screening** with calibrated risk + SHAP/LIME concordance — without claiming diagnosis or causation.

## 4. Dataset
CDC BRFSS2015 Diabetes Binary — 253,680 rows, 21 features, ~86:14 imbalance

## 5. Method
Stratified 70/15/15 · train-only SMOTE · F1 tuning · validation-tuned thresholds · test sacred

## 6. Confirmed Phase-1 models
LR · Random Forest · XGBoost · LightGBM · MLP

## 7. Official test results
| Model | Recall | F1 | ROC-AUC | Precision | Threshold |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.670 | **0.462** | **0.824** | 0.353 | 0.59 |
| MLP | 0.643 | 0.457 | 0.820 | 0.354 | 0.54 |
| LightGBM | 0.651 | 0.455 | 0.818 | 0.350 | 0.38 |
| Random Forest | 0.619 | 0.455 | 0.814 | 0.360 | 0.50 |
| XGBoost | 0.642 | 0.453 | 0.809 | 0.350 | 0.48 |

**Selected:** Logistic Regression (best F1 + ROC-AUC; fast; interpretable).  
Do **not** pick by accuracy alone — FN burden matters.

Figures: `results/phase1/figures/`

## 8. Explainability
SHAP + LIME on promoted LR; concordance ≠ certainty; not causal

## 9. Ethics
False reassurance · misleading explanations · no causality · US→India transfer limits · calibration for % risk

## 10. System
Flutter → FastAPI → model + explanations (Firebase planned)

## 11. Next
Flutter SDK run · Firebase optional · AWS API deploy optional · local clinical validation plan
