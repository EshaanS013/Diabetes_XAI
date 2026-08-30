# Phase 1 Experiment Report

**Generated:** 2026-08-30T10:20:02.057234+00:00
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
              model          display_name    status  accuracy  precision   recall  specificity       f1  roc_auc   pr_auc  brier_score      ece  cv_best_score  training_time  inference_ms_per_row  decision_threshold  random_seed  calibrated metrics_split
logistic_regression   Logistic Regression completed  0.782855   0.352935 0.670124     0.801105 0.462359 0.824479 0.422363     0.173831 0.237420       0.443380      50.597260              0.001013                0.59           42        True          test
      random_forest         Random Forest completed  0.793499   0.359868 0.619012     0.821746 0.455138 0.813648 0.405164     0.143159 0.188063       0.450960    2094.305771              0.004164                0.50           42        True          test
            xgboost               XGBoost completed  0.781174   0.346928 0.646548     0.802968 0.451558 0.808016 0.402289     0.148237 0.209688       0.445093     424.486106              0.001534                0.48           42        True          test
           lightgbm              LightGBM completed  0.782882   0.349959 0.651075     0.804220 0.455229 0.817941 0.416741     0.118957 0.135402       0.434352     355.037188              0.002256                0.38           42        True          test
                mlp Multilayer Perceptron completed  0.787034   0.354366 0.642965     0.810357 0.456909 0.819551 0.416340     0.151148 0.186369       0.446966    2418.565701              0.000862                0.54           42        True          test
```

### Model selection

**Selected model:** `logistic_regression`

**Justification:**
- Highest F1 on held-out test (0.4624)
- Competitive recall (0.6701); highest recall: `logistic_regression` (0.6701)
- ROC-AUC: 0.8245 (leader: `logistic_regression` 0.8245)
- Do **not** select on accuracy alone; false negatives matter in screening.

## Failures

No recorded failures in latest `test_metrics.json`.

## Ethics & limitations (must appear in paper/presentation)

- False reassurance risk from false negatives
- Misleading explanations / disagreement between SHAP and LIME
- SHAP/LIME are not causal
- US survey population may not transfer to Indian clinical use without local validation
- Probability displays require calibration checks

