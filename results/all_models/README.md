# Ten-model benchmark

Models are ranked by cross-validation F1. Test-set values are final evaluation only.

| cv_rank | model | benchmark_phase | cv_best_score | f1 | recall | precision | roc_auc | pr_auc | calibrated_brier_score | calibrated_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | random_forest | phase1 | 0.4510 | 0.4551 | 0.6190 | 0.3599 | 0.8136 | 0.4052 | not recorded | not recorded |
| 2 | mlp | phase1 | 0.4470 | 0.4569 | 0.6430 | 0.3544 | 0.8196 | 0.4163 | not recorded | not recorded |
| 3 | xgboost | phase1 | 0.4451 | 0.4516 | 0.6465 | 0.3469 | 0.8080 | 0.4023 | not recorded | not recorded |
| 4 | catboost | phase2 | 0.4444 | 0.4521 | 0.5952 | 0.3645 | 0.8117 | 0.4036 | 0.0995 | 0.0057 |
| 5 | logistic_regression | phase1 | 0.4434 | 0.4624 | 0.6701 | 0.3529 | 0.8245 | 0.4224 | not recorded | not recorded |
| 6 | gradient_boosting | phase2 | 0.4418 | 0.4510 | 0.5577 | 0.3785 | 0.8138 | 0.4088 | 0.0991 | 0.0030 |
| 7 | svm | phase2 | 0.4411 | 0.4630 | 0.5947 | 0.3790 | 0.8243 | 0.4224 | 0.0978 | 0.0052 |
| 8 | lightgbm | phase1 | 0.4344 | 0.4552 | 0.6511 | 0.3500 | 0.8179 | 0.4167 | not recorded | not recorded |
| 9 | naive_bayes | phase2 | 0.4077 | 0.4264 | 0.6409 | 0.3195 | 0.7805 | 0.3519 | 0.1038 | 0.0032 |
| 10 | knn | phase2 | 0.4069 | 0.4261 | 0.6662 | 0.3132 | 0.7866 | 0.3470 | 0.1034 | 0.0044 |

## Selection guardrail

The `test_f1_rank_retrospective` column must not be used for model selection. Clinical utility, subgroup analysis, calibration, latency, and XAI review remain required before promotion.
