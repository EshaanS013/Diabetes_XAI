from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import load_schema
from src.data.split import assert_disjoint, make_stratified_splits
from src.evaluation.metrics import classification_metrics, specificity_score
from src.explainability.generate import agreement_metrics
from src.models.registry import list_models
from src.preprocessing.pipeline import build_model_pipeline, BMIIQRCapper
from sklearn.linear_model import LogisticRegression


def test_schema_loads_21_features():
    schema = load_schema()
    assert schema.n_features == 21
    assert schema.target == "Diabetes_binary"
    assert set(schema.allowed_target_values) == {0, 1}


def test_registry_has_ten_models():
    models = list_models()
    assert len(models) == 10
    for required in [
        "logistic_regression",
        "naive_bayes",
        "knn",
        "svm",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
    ]:
        assert required in models


def test_specificity_and_metrics_known_arrays():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.4, 0.6, 0.3])
    y_pred = (y_prob >= 0.5).astype(int)
    # TN=3, FP=1, FN=0, TP=4 at 0.5? Let's compute carefully:
    # pairs: (0,0),(0,0),(1,1),(1,1),(1,1),(0,0),(1,1),(0,0) if thresh 0.5
    # y_pred: 0,0,1,1,1,0,1,0
    # TN: indices 0,1,5,7 => 4; FP: none; FN: none; TP: 2,3,4,6 => 4
    assert specificity_score(y_true, y_pred) == pytest.approx(1.0)
    m = classification_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["confusion_matrix"]["fn"] == 0


def test_splits_are_disjoint_and_cover_all():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=1000))
    splits = make_stratified_splits(y, random_seed=42)
    assert_disjoint(splits)
    all_idx = set(map(int, np.concatenate(list(splits.values()))))
    assert all_idx == set(range(1000))
    n = 1000
    assert abs(len(splits["train"]) / n - 0.70) < 0.02
    assert abs(len(splits["validation"]) / n - 0.15) < 0.02
    assert abs(len(splits["test"]) / n - 0.15) < 0.02


def test_bmi_capper_fits_train_only_semantics():
    train = pd.DataFrame({"BMI": [20, 22, 24, 26, 28, 80]})
    test = pd.DataFrame({"BMI": [10, 90]})
    cap = BMIIQRCapper(multiplier=1.5).fit(train)
    out = cap.transform(test)
    assert out["BMI"].min() >= cap.lower_
    assert out["BMI"].max() <= cap.upper_


def test_smote_pipeline_does_not_require_val_in_fit():
    """SMOTE lives inside training pipeline; fitting on train alone must succeed."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "BMI": rng.normal(28, 5, 200),
            "GenHlth": rng.integers(1, 6, 200),
            "Age": rng.integers(1, 14, 200),
            "HighBP": rng.integers(0, 2, 200),
            "HighChol": rng.integers(0, 2, 200),
            "CholCheck": rng.integers(0, 2, 200),
            "Smoker": rng.integers(0, 2, 200),
            "Stroke": rng.integers(0, 2, 200),
            "HeartDiseaseorAttack": rng.integers(0, 2, 200),
            "PhysActivity": rng.integers(0, 2, 200),
            "Fruits": rng.integers(0, 2, 200),
            "Veggies": rng.integers(0, 2, 200),
            "HvyAlcoholConsump": rng.integers(0, 2, 200),
            "AnyHealthcare": rng.integers(0, 2, 200),
            "NoDocbcCost": rng.integers(0, 2, 200),
            "MentHlth": rng.integers(0, 31, 200),
            "PhysHlth": rng.integers(0, 31, 200),
            "DiffWalk": rng.integers(0, 2, 200),
            "Sex": rng.integers(0, 2, 200),
            "Education": rng.integers(1, 7, 200),
            "Income": rng.integers(1, 9, 200),
        }
    )
    y = pd.Series(rng.integers(0, 2, 200))
    # Ensure both classes present
    y.iloc[:20] = 1
    y.iloc[20:40] = 0
    pipe = build_model_pipeline(
        LogisticRegression(max_iter=500),
        scale=True,
        use_smote=True,
        smote_k_neighbors=3,
    )
    pipe.fit(X, y)
    pred = pipe.predict_proba(X.iloc[:5])
    assert pred.shape == (5, 2)


def test_agreement_metrics_semantics():
    shap = {"BMI": 0.5, "Age": 0.4, "HighBP": 0.1}
    lime = {"BMI": 0.2, "GenHlth": 0.3, "Age": 0.1}
    out = agreement_metrics(shap, lime, top_k=2)
    assert "BMI" in out["shap_top_features"]
    assert out["overlap_count"] >= 1
    assert "NOT predictive confidence" in out["note"]
