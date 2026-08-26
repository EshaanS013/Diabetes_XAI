# Phase 1 Experiment Report

**Generated:** 2026-08-26T16:12:04.124620+00:00
**Model selection status:** `confirmed`

## Important

- This system is a **screening aid**, not a diagnostic device.
- Metrics below are included **only if measured**. Missing values are marked TBD.
- False negatives matter in healthcare screening; do not select on accuracy alone.
- SHAP/LIME agreement is explanation concordance, **not** predictive confidence.

## Configured Phase-1 models

- `logistic_regression`
- `random_forest`
- `xgboost`
- `lightgbm`
- `mlp`

**Rationale (proposed):** Confirmed five-model Phase-1 suite: Logistic Regression (linear baseline), Random Forest (bagging), XGBoost (proposal primary booster), LightGBM (second booster), MLP (non-tree neural). Architecture still supports all ten registered algorithms for later phases.

## Results

```
              model          display_name    status  accuracy  precision   recall  specificity       f1  roc_auc   pr_auc  brier_score      ece  cv_best_score  training_time  inference_ms_per_row  decision_threshold  random_seed  calibrated metrics_split         source_experiment
           lightgbm              LightGBM completed  0.782882   0.349959 0.651075     0.804220 0.455229 0.817941 0.416741     0.118957 0.135402       0.434352     197.307989              0.001364                0.38           42        True          test 20260825T171739Z_4aa740a6
logistic_regression   Logistic Regression completed  0.782882   0.352941 0.669936     0.801166 0.462319 0.824479 0.422357     0.173827 0.237441       0.443380      19.041758              0.000319                0.59           42        True          test 20260825T171739Z_4aa740a6
                mlp Multilayer Perceptron completed  0.787034   0.354366 0.642965     0.810357 0.456909 0.819551 0.416340     0.151148 0.186369       0.446966    2027.015489              0.000570                0.54           42        True          test 20260826T152829Z_81d2eeac
      random_forest         Random Forest completed  0.793499   0.359868 0.619012     0.821746 0.455138 0.813648 0.405164     0.143159 0.188063       0.450960     555.509748              0.001804                0.50           42        True          test 20260825T171739Z_4aa740a6
            xgboost               XGBoost completed  0.784196   0.350247 0.641833     0.807243 0.453190 0.808575 0.403803     0.147349 0.207948       0.444278     172.951350              0.000476                0.48           42        True          test 20260825T171739Z_4aa740a6
```

### Model selection discussion (template)

Compare models on recall, F1, ROC-AUC, precision, specificity, calibration,
latency, and explainability compatibility. Do **not** auto-declare a winner
from a single metric. Fill after reviewing measured results.

**Tentative selected model:** TBD - generated after experimental run

**Justification:** TBD - generated after experimental run

## Failures

No recorded failures in latest `test_metrics.json`.

## Ethics & limitations (must appear in paper/presentation)

- False reassurance risk from false negatives
- Misleading explanations / disagreement between SHAP and LIME
- SHAP/LIME are not causal
- US survey population may not transfer to Indian clinical use without local validation
- Probability displays require calibration checks

