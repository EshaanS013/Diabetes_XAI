# Ethics, Explainability Limits, and Clinical-Use Constraints

This document feeds the research paper ethics section. It is mandatory for the Phase-1 panel.

## Intended use

Preliminary **diabetes risk screening aid** for research / decision-support prototyping.  
**Not** a diagnostic device. **Not** a substitute for clinical judgment.

## False reassurance (false negatives)

A false negative may lead a user to believe they need no further attention. In screening, false-negative burden is ethically material. Model selection must weigh **recall/sensitivity**, not only accuracy.

## False alarm burden (false positives)

False positives can cause anxiety and unnecessary care-seeking. Precision and specificity remain relevant; tradeoffs must be disclosed.

## Misleading explanations

SHAP and LIME can disagree. Disagreement is **explanation-method concordance**, not proof that the prediction is wrong or right. Never present agreement as “model certainty” or calibrated confidence.

## Causality

Feature attributions describe **how the model used features**, not causal effects on disease. Do not use causal language (“caused”, “leads to disease”) in patient-facing copy unless causal methods were used (they were not).

## SHAP / LIME limitations

- SHAP: computational cost; assumptions of feature independence in some estimators; unstable with correlated features
- LIME: local surrogate approximation; stochastic; stability must be measured
- Neither validates clinical truth
- Templates mapping features → text require clinical review (`clinical_review_status: pending`)

## Population / transfer risk

BRFSS2015 reflects a **US survey population**. Results must **not** be claimed as validated for Indian clinical populations without local data and clinical validation.

## Calibration & risk percentages

Showing “73% risk” implies probability quality. Report Brier/ECE/calibration curves. Prefer calibrated probabilities for user-facing percentages.

## Privacy

Minimize logging of raw health features. No secrets in git. Do not claim HIPAA/GDPR/Indian-law compliance without formal assessment.

## Dual audience

Patient view: plain language, non-alarmist, strong disclaimer.  
Doctor view: fuller metrics, concordance flags, model/version metadata — still not a diagnosis.
