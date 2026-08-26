# Model Selection Report (Phase 1 — Official)

**Generated after official full-data run.**  
**Selection status:** confirmed  

## Policy
- Do not select on accuracy alone.
- Weigh recall/FN, F1, ROC-AUC, calibration, latency, explainability.

## Official test leaders
- Highest F1 + ROC-AUC: `logistic_regression` (F1=0.462, AUC=0.824, recall=0.670)
- Highest recall among close contenders: `lightgbm` (recall=0.651) with slightly lower F1

## Selected model
**logistic_regression** (`phase1-official-lr-v1`)

Justification:
- Best F1 and ROC-AUC on held-out test
- Competitive recall
- Fast inference
- Compatible with coefficient + SHAP/LIME explanations

Caveats: US survey population; screening aid only; threshold not clinically validated; SHAP/LIME concordance is not confidence.
