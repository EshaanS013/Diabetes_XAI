# Phase 1 Experiment Report

**Generated:** 2026-08-25T17:06:35.795771+00:00
**Model selection status:** `proposed_pending_confirmation`

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

**Rationale (proposed):** Span linear, bagging, boosting (proposal primary), second booster (family comparison), and neural net (non-tree family). Keeps proposal's LR/RF/XGBoost intact and adds LightGBM + MLP for Phase-1 breadth without claiming mentor selection.

## Results

```
              model          display_name    status  accuracy  precision   recall  specificity       f1  roc_auc   pr_auc  brier_score      ece  cv_best_score  training_time  inference_ms_per_row  decision_threshold  random_seed  calibrated metrics_split
logistic_regression   Logistic Regression completed  0.782750   0.352707 0.669559     0.801075 0.462029 0.824347 0.422473     0.173295 0.234369       0.445055       7.594705              0.000327                0.59           42        True          test
      random_forest         Random Forest completed  0.798965   0.366439 0.607507     0.829959 0.457139 0.817297 0.408797     0.122271 0.129404       0.436768      17.712300              0.003223                0.44           42        True          test
            xgboost               XGBoost completed  0.788979   0.357083 0.642776     0.812647 0.459114 0.821547 0.423823     0.112799 0.117122       0.414538      11.450759              0.000777                0.36           42        True          test
           lightgbm              LightGBM completed  0.784564   0.351761 0.648057     0.806662 0.456005 0.818040 0.418483     0.114679 0.121199       0.416481      10.368207              0.001380                0.36           42        True          test
                mlp Multilayer Perceptron completed  0.788348   0.348758 0.598453     0.819089 0.440694 0.803798 0.387074     0.135373 0.132164       0.438469      89.141075              0.000543                0.48           42        True          test
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

