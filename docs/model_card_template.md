# Model Card (Template)

Fill after a model is selected. Never invent metrics.

| Field | Value |
|---|---|
| Model name | TBD |
| Model version | TBD |
| Algorithm | TBD |
| Dataset | BRFSS2015 diabetes binary |
| Population | US adults (survey) |
| Features | 21 (see schema) |
| Target | Diabetes_binary |
| Split | 70/15/15 stratified, seed 42 |
| Training date | TBD |
| Random seed | 42 |
| Hyperparameters | TBD |
| Evaluation metrics | TBD - generated after experimental run |
| Calibration | TBD |
| Explainability | SHAP + LIME |
| Intended use | Preliminary screening aid |
| Out of scope | Diagnosis, treatment, emergency triage |
| Known limitations | See docs/ethics.md |
| Fairness considerations | Population shift; self-report bias; demographic encoding |
| Ethical considerations | FN risk; explanation misuse; pending clinical template review |
| Deployment | FastAPI artifact + Flutter client |

**Not a diagnostic tool.**
