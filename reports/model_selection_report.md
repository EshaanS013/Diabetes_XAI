# Model Selection Report (Phase 1)

**Generated:** 2026-08-30T10:20:02.870266+00:00
**Selection status:** `confirmed`

## Policy

- Do **not** select on accuracy alone.
- Prioritize recall/FN burden, F1, ROC-AUC, calibration, latency, explainability fit.
- Screening aid only — not a diagnostic claim.

## Measured comparison

```
              model          display_name    status  accuracy  precision   recall  specificity       f1  roc_auc   pr_auc  brier_score      ece  cv_best_score  training_time  inference_ms_per_row  decision_threshold  random_seed  calibrated metrics_split
logistic_regression   Logistic Regression completed  0.782855   0.352935 0.670124     0.801105 0.462359 0.824479 0.422363     0.173831 0.237420       0.443380      50.597260              0.001013                0.59           42        True          test
      random_forest         Random Forest completed  0.793499   0.359868 0.619012     0.821746 0.455138 0.813648 0.405164     0.143159 0.188063       0.450960    2094.305771              0.004164                0.50           42        True          test
            xgboost               XGBoost completed  0.781174   0.346928 0.646548     0.802968 0.451558 0.808016 0.402289     0.148237 0.209688       0.445093     424.486106              0.001534                0.48           42        True          test
           lightgbm              LightGBM completed  0.782882   0.349959 0.651075     0.804220 0.455229 0.817941 0.416741     0.118957 0.135402       0.434352     355.037188              0.002256                0.38           42        True          test
                mlp Multilayer Perceptron completed  0.787034   0.354366 0.642965     0.810357 0.456909 0.819551 0.416340     0.151148 0.186369       0.446966    2418.565701              0.000862                0.54           42        True          test
```

- Highest **recall**: `logistic_regression` (recall=0.6701) — not automatically the production choice.
- Highest **f1**: `logistic_regression` (f1=0.4624) — not automatically the production choice.
- Highest **roc_auc**: `logistic_regression` (roc_auc=0.8245) — not automatically the production choice.

## Selected model

**logistic_regression** — highest measured F1 (0.4624) and competitive recall (0.6701), ROC-AUC (0.8245).

**Justification checklist:**
- [ ] Recall / false-negative burden acceptable for screening context
- [ ] F1 competitive under imbalance
- [ ] ROC-AUC / PR-AUC reviewed
- [ ] Calibration (Brier/ECE) acceptable for % risk display
- [ ] Inference latency acceptable for mobile
- [ ] Explainability path (TreeSHAP vs kernel) feasible
- [ ] Stability across CV folds reviewed

## Caveats

- Official publication requires `selection_status: confirmed` (currently `confirmed`).
- Development/subsample runs must be labelled as such in the paper/panel.

