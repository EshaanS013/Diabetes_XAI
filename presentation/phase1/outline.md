# Phase 1 Panel Presentation Outline

**Timing target:** first week of September  
**Rule:** only show measured numbers; otherwise say “experiment pending / TBD”.

## Slide 1 — Title

Enhancing Clinical Decision Support through Explainable AI  
Diabetes Risk Prediction (Screening Aid — Not Diagnosis)

## Slide 2 — Problem

Black-box risk scores are hard to trust. Need prediction **and** why.

## Slide 3 — Dataset choice

BRFSS2015 vs PIMA: self-reportable features, scale, mobile fit.

## Slide 4 — Methodology

70/15/15 stratified; anti-leakage; SMOTE train-only; F1-oriented tuning.

## Slide 5 — Five models

List confirmed five (or clearly label proposed pending confirmation).

## Slide 6 — Results

Table from `results/phase1/model_comparison.csv`  
Emphasize recall / FN tradeoffs — not accuracy alone.

## Slide 7 — Explainability

SHAP + LIME; concordance demo; **not certainty**.

## Slide 8 — Ethics

False reassurance; misleading explanations; non-causality; India transfer limits.

## Slide 9 — System

Flutter → FastAPI → model + explanations; screening disclaimer.

## Slide 10 — Next steps

Confirm models if needed; freeze candidate; mobile polish; local validation plan.

## Anticipated questions (prep)

- Why both SHAP and LIME? What if they disagree?  
- Can SHAP prove causality?  
- Can the system diagnose diabetes?  
- Why might US results not transfer to India?  
- Why FastAPI + Flutter?
