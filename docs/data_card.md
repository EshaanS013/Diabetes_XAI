# Data Card — CDC BRFSS2015 Diabetes Binary Health Indicators

| Field | Value |
|---|---|
| Name | Diabetes Binary Health Indicators (BRFSS2015 cleaned extract) |
| Source | CDC Behavioral Risk Factor Surveillance System 2015 |
| Access | UCI ML Repository / Kaggle community extracts |
| Year | 2015 survey year |
| Approx. rows | ~253,680 |
| Features | 21 |
| Target | `Diabetes_binary` (0 = no diabetes; 1 = prediabetes or diabetes) |
| Class balance | ~86% negative / ~14% positive (verify on local file) |
| Population | US adults, both sexes, broad age range |
| Feature type | Self-reportable survey indicators (no lab glucose/insulin required) |

## Why BRFSS over PIMA

PIMA is small, female-only in the classic extract, and depends on laboratory values unsuitable for a self-service mobile questionnaire. BRFSS aligns with the mobile screening product goal.

## Limitations

- Self-report bias and survey coding artifacts
- US-centric; not automatically generalizable to India
- Binary target collapses prediabetes and diabetes
- Social determinants encoded coarsely (income, education)

## Pipeline notes

- Dataset hash recorded at prepare-time
- Missing-value audit always run (cleaned extract expected to have none)
- BMI IQR capping thresholds learned on train only
- Duplicates audited (not silently dropped unless justified)

## License / redistribution

Follow the upstream dataset license/terms. Large CSVs are **not** committed to git by default (`data/raw/` gitignored except docs).
