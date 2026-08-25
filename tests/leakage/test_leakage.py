from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.split import assert_disjoint, make_stratified_splits
from src.preprocessing.pipeline import build_preprocessor


def test_no_split_overlap():
    y = pd.Series([0] * 700 + [1] * 300)
    splits = make_stratified_splits(y, random_seed=7)
    assert_disjoint(splits)


def test_preprocessor_fit_transform_shapes():
    schema_feats = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
    ]
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame({c: rng.integers(0, 2, 50) for c in schema_feats})
    X_train["BMI"] = rng.normal(28, 4, 50)
    X_train["GenHlth"] = rng.integers(1, 6, 50)
    X_train["MentHlth"] = rng.integers(0, 31, 50)
    X_train["PhysHlth"] = rng.integers(0, 31, 50)
    X_train["Age"] = rng.integers(1, 14, 50)
    X_train["Education"] = rng.integers(1, 7, 50)
    X_train["Income"] = rng.integers(1, 9, 50)
    X_val = X_train.copy()

    pre = build_preprocessor(scale=True, bmi_iqr_cap=True)
    pre.fit(X_train)
    Xt = pre.transform(X_val)
    assert Xt.shape[0] == len(X_val)
    assert Xt.shape[1] == 21
