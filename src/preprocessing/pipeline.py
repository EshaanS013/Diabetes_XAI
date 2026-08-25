from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.schema import DataSchema, load_schema


CONTINUOUS_LIKE = {"continuous", "continuous_count"}


def feature_groups(schema: DataSchema | None = None) -> tuple[list[str], list[str]]:
    schema = schema or load_schema()
    continuous: list[str] = []
    categorical: list[str] = []
    for name, meta in schema.features.items():
        dtype = meta.get("dtype", "binary")
        if dtype in CONTINUOUS_LIKE:
            continuous.append(name)
        else:
            categorical.append(name)
    return continuous, categorical


class BMIIQRCapper(BaseEstimator, TransformerMixin):
    """IQR-based BMI capping; thresholds fitted on training data only."""

    def __init__(self, column: str = "BMI", multiplier: float = 1.5):
        self.column = column
        self.multiplier = multiplier

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "BMIIQRCapper":
        df = self._as_df(X)
        if self.column not in df.columns:
            self.lower_ = None
            self.upper_ = None
            return self
        q1 = float(df[self.column].quantile(0.25))
        q3 = float(df[self.column].quantile(0.75))
        iqr = q3 - q1
        self.lower_ = q1 - self.multiplier * iqr
        self.upper_ = q3 + self.multiplier * iqr
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        df = self._as_df(X).copy()
        if self.column in df.columns and getattr(self, "lower_", None) is not None:
            df[self.column] = df[self.column].clip(self.lower_, self.upper_)
        return df

    @staticmethod
    def _as_df(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("BMIIQRCapper expects a pandas DataFrame with named columns")


def build_column_transformer(
    *,
    scale: bool,
    schema: DataSchema | None = None,
) -> ColumnTransformer:
    schema = schema or load_schema()
    continuous, categorical = feature_groups(schema)
    transformers: list[tuple] = []
    if continuous:
        cont_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            cont_steps.append(("scaler", StandardScaler()))
        transformers.append(("continuous", Pipeline(cont_steps), continuous))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_preprocessor(
    *,
    scale: bool,
    schema: DataSchema | None = None,
    bmi_iqr_cap: bool = True,
    bmi_iqr_multiplier: float = 1.5,
) -> Pipeline:
    """Sklearn Pipeline for non-SMOTE paths (nested Pipeline OK here)."""
    steps: list[tuple] = []
    if bmi_iqr_cap:
        steps.append(("bmi_cap", BMIIQRCapper(column="BMI", multiplier=bmi_iqr_multiplier)))
    steps.append(("columns", build_column_transformer(scale=scale, schema=schema)))
    return Pipeline(steps)


def build_model_pipeline(
    estimator: Any,
    *,
    scale: bool,
    use_smote: bool,
    smote_random_state: int = 42,
    smote_k_neighbors: int = 5,
    schema: DataSchema | None = None,
    bmi_iqr_cap: bool = True,
    bmi_iqr_multiplier: float = 1.5,
) -> Any:
    """
    Leakage-safe training pipeline.

    Steps are kept flat because imblearn.Pipeline forbids nested Pipelines
    as intermediate steps.
    """
    steps: list[tuple] = []
    if bmi_iqr_cap:
        steps.append(("bmi_cap", BMIIQRCapper(column="BMI", multiplier=bmi_iqr_multiplier)))
    steps.append(("columns", build_column_transformer(scale=scale, schema=schema)))
    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=smote_random_state,
                    k_neighbors=smote_k_neighbors,
                ),
            )
        )
        steps.append(("model", estimator))
        return ImbPipeline(steps=steps)
    steps.append(("model", estimator))
    return Pipeline(steps=steps)
