# Phase 1 Panel — Slide Content Draft

**Status:** Includes **measured development** metrics (25k stratified train subsample).  
**Do not present these as final full-data / official Phase-1 results.**

---

## 1. Title
**Enhancing Clinical Decision Support through Explainable AI**  
Interpretable ML for Diabetes Risk Prediction  
*Screening aid — not a diagnosis*

## 2. Motivation
- India’s diabetes burden is large; many cases undiagnosed (IDF Atlas context in proposal)
- Black-box scores lack actionable “why”
- Need mobile-ready, self-reportable inputs (no lab glucose)

## 3. Research question
Can we build a **leakage-free**, explainable diabetes **risk screening** pipeline that returns:
1. a calibrated risk estimate, and  
2. dual explanations (SHAP + LIME) with concordance — without claiming diagnosis or causation?

## 4. Dataset
- CDC BRFSS2015 Diabetes Binary Health Indicators
- 253,680 rows, 21 features, ~86:14 imbalance (`reports/data_quality_report.md`)
- Chosen over PIMA: self-reportable + larger + both sexes

## 5. Method (anti-leakage)
- Stratified 70 / 15 / 15
- Preprocess + SMOTE **train/CV-train only**
- Tune on F1; validation-tuned decision threshold; test sacred
- Metrics: accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, Brier, ECE

## 6. Phase-1 models
**Proposed (pending confirmation):** LR, Random Forest, XGBoost, LightGBM, MLP  
Codebase also registers NB, k-NN, SVM, GB, CatBoost for later phases.

## 7. Results (development benchmark — labelled)
| Model | Recall | F1 | ROC-AUC | Precision | Threshold |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.656 | **0.465** | **0.824** | 0.360 | 0.60 |
| MLP | 0.627 | 0.456 | 0.819 | 0.358 | 0.58 |
| XGBoost | 0.624 | 0.454 | 0.818 | 0.356 | 0.34 |
| Random Forest | 0.643 | 0.449 | 0.815 | 0.345 | 0.38 |
| LightGBM | **0.658** | 0.448 | 0.817 | 0.340 | 0.32 |

**Takeaway for panel:** do **not** pick by accuracy alone. On this dev run LR leads F1/AUC; LightGBM leads recall. Final pick needs full-data + calibration + latency + XAI fit.

Figures: `results/phase1/figures/`

## 8. Explainability
- SHAP + LIME on promoted LR (dev)
- Mean Jaccard top-3 ≈ **0.66**; exact top-3 agreement ≈ **0.47** (n=15)
- Concordance = explanation agreement, **not** certainty
- Templates pending clinical review

## 9. Ethics (supervisor-required)
- False reassurance (FN) / false alarm (FP)
- Misleading explanations; SHAP≠LIME≠truth
- No causality claims
- US survey ≠ validated Indian clinical use
- Probability % needs calibration checks

## 10. System architecture
Flutter (patient/doctor) → FastAPI → model + explanations → Firebase auth/history (planned)  
API running path: `/health`, `/model-info`, `/predict`

## 11. Next steps
1. Confirm five algorithms with supervisor  
2. Full-data official Phase-1 run  
3. Freeze production model + signed model card  
4. Install Flutter SDK and connect client to `/predict`

## Backup Q&A
- Why BRFSS not PIMA?  
- Why both SHAP and LIME?  
- What if they disagree?  
- Can this diagnose diabetes? **No.**
