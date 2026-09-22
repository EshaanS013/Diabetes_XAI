# Phase 1 Panel — Slide Content Draft

**Status:** Official full-data Phase-1 metrics (confirmed five models).  
**Presentation date:** 3 September 2026  
**Repo:** https://github.com/EshaanS013/Diabetes_XAI

---

## Slide 1 — Title
**Enhancing Clinical Decision Support through Explainable AI**  
Interpretable ML for Diabetes Risk Prediction  
*Screening aid — not a diagnosis*

## Slide 2 — Supervisor milestone (Phase 1)
- Train & compare **5 ML algorithms**
- Report accuracy, F1, recall, ROC-AUC, ROC curves
- Research paper draft with **ethics & explainability limits**
- Panel presentation (September 2026)

## Slide 3 — Motivation
- India’s diabetes burden is large; many cases undiagnosed
- Black-box scores lack actionable “why”
- Need mobile-ready, self-reportable inputs (no lab glucose)

## Slide 4 — Research question
Leakage-free explainable **risk screening** with calibrated risk + SHAP/LIME concordance — without claiming diagnosis or causation.

## Slide 5 — Dataset
CDC BRFSS2015 Diabetes Binary — **253,680** rows, **21** features, **~86:14** imbalance  
Self-reportable features suitable for mobile questionnaire

## Slide 6 — Method
Stratified **70/15/15** · train-only SMOTE · **F1 tuning** · validation-tuned thresholds · test sacred

## Slide 7 — Confirmed Phase-1 models
1. Logistic Regression  
2. Random Forest  
3. XGBoost  
4. LightGBM  
5. MLP  

## Slide 8 — Official test results
| Model | Accuracy | Recall | F1 | ROC-AUC | Precision |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.783 | **0.670** | **0.462** | **0.824** | 0.353 |
| MLP | 0.787 | 0.643 | 0.457 | 0.820 | 0.354 |
| LightGBM | 0.783 | 0.651 | 0.455 | 0.818 | 0.350 |
| Random Forest | 0.793 | 0.619 | 0.455 | 0.814 | 0.360 |
| XGBoost | 0.781 | 0.647 | 0.452 | 0.808 | 0.347 |

**Selected:** Logistic Regression (best F1 + ROC-AUC; fast; interpretable).  
Do **not** pick by accuracy alone — FN burden matters.

## Slide 9 — ROC curves
Show `results/phase1/figures/roc_curves.png`  
Explain TPR vs FPR; AUC summarises ranking, not clinical utility alone

## Slide 10 — Confusion matrix (selected LR)
From test split at threshold 0.59 — discuss FN vs FP tradeoff  
Figure: `results/phase1/figures/confusion_matrices/logistic_regression.png`

## Slide 11 — Explainability
SHAP + LIME on promoted LR; mean top-3 Jaccard **0.42**; exact agreement **~3%**  
Concordance ≠ certainty; not causal

## Slide 12 — Ethics (required by supervisor)
- False reassurance from false negatives  
- Misleading / unstable explanations (LIME)  
- SHAP/LIME are not causal  
- US BRFSS → India transfer limits  
- Calibration required for % risk display  
- **Not a diagnostic device**

## Slide 13 — System architecture (Phase 2+)
Flutter → FastAPI → model + explanations (AWS deploy optional)

## Slide 14 — Next steps
- Flutter mobile UI with questionnaire  
- AWS student account for API hosting (optional)  
- Local clinical validation plan (India)  
- Remaining 5 algorithms in Phase 2

## Anticipated panel questions
- Why BRFSS not PIMA? → scale, self-reportable, both sexes  
- Why these five models? → span linear, bagging, boosting×2, neural  
- Can SHAP prove causality? → **No**  
- Can it diagnose? → **No — screening aid only**  
- Why US data for Indian app? → proxy features; local validation required  
- Why LR over Random Forest (higher accuracy)? → recall/F1/AUC tradeoff for screening
