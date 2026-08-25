from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.schema import load_schema
from src.explainability.generate import agreement_metrics, render_safe_explanations
from src.utils.config import PROJECT_ROOT, setup_logging

logger = setup_logging()

DISCLAIMER = (
    "This output is a preliminary diabetes risk screening aid based on a statistical model. "
    "It is NOT a medical diagnosis. Explanations reflect model associations, not proven causation. "
    "Consult a qualified clinician for medical advice."
)


class HealthFeatures(BaseModel):
    HighBP: float = Field(..., ge=0, le=1)
    HighChol: float = Field(..., ge=0, le=1)
    CholCheck: float = Field(..., ge=0, le=1)
    BMI: float = Field(..., ge=12, le=100)
    Smoker: float = Field(..., ge=0, le=1)
    Stroke: float = Field(..., ge=0, le=1)
    HeartDiseaseorAttack: float = Field(..., ge=0, le=1)
    PhysActivity: float = Field(..., ge=0, le=1)
    Fruits: float = Field(..., ge=0, le=1)
    Veggies: float = Field(..., ge=0, le=1)
    HvyAlcoholConsump: float = Field(..., ge=0, le=1)
    AnyHealthcare: float = Field(..., ge=0, le=1)
    NoDocbcCost: float = Field(..., ge=0, le=1)
    GenHlth: float = Field(..., ge=1, le=5)
    MentHlth: float = Field(..., ge=0, le=30)
    PhysHlth: float = Field(..., ge=0, le=30)
    DiffWalk: float = Field(..., ge=0, le=1)
    Sex: float = Field(..., ge=0, le=1)
    Age: float = Field(..., ge=1, le=13)
    Education: float = Field(..., ge=1, le=6)
    Income: float = Field(..., ge=1, le=8)


class PredictRequest(BaseModel):
    features: HealthFeatures
    include_explanations: bool = False
    audience: str = Field(default="patient", pattern="^(patient|doctor)$")


class PredictResponse(BaseModel):
    risk_probability: float
    risk_percent: float
    predicted_label: int
    threshold: float
    model_name: str | None
    model_version: str
    disclaimer: str
    explanations: dict[str, Any] | None = None


def _artifact_path() -> Path:
    env = os.getenv("MODEL_ARTIFACT_PATH", "artifacts/production/model.joblib")
    path = Path(env)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@lru_cache(maxsize=1)
def load_production_artifact() -> dict[str, Any]:
    path = _artifact_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Production model not found at {path}. "
            "Train models and copy the selected artifact to artifacts/production/model.joblib"
        )
    return joblib.load(path)


def _ablation_contributions(model: Any, row: pd.DataFrame, baseline_proba: float) -> dict[str, float]:
    """
    Fast local attribution: Δp when each feature is replaced by its column median-ish
    baseline of 0 for binary / mean of provided value scale.
    Association only — not SHAP/LIME and not causal.
    """
    scores: dict[str, float] = {}
    for col in row.columns:
        perturbed = row.copy()
        # Use 0 as reference for most BRFSS binary/ordinal fields; BMI -> 25 (approx healthy)
        if col == "BMI":
            perturbed.loc[:, col] = 25.0
        elif col in {"GenHlth"}:
            perturbed.loc[:, col] = 3.0
        elif col in {"MentHlth", "PhysHlth"}:
            perturbed.loc[:, col] = 0.0
        elif col in {"Age", "Education", "Income"}:
            perturbed.loc[:, col] = float(row.iloc[0][col])  # leave unchanged if ordinal-heavy
            # Instead: shift toward mid category
            mids = {"Age": 7.0, "Education": 4.0, "Income": 5.0}
            perturbed.loc[:, col] = mids[col]
        else:
            perturbed.loc[:, col] = 0.0
        p = float(model.predict_proba(perturbed)[0, 1])
        scores[col] = float(baseline_proba - p)
    return scores


def _lime_contributions(model: Any, row: pd.DataFrame, feature_order: list[str]) -> dict[str, float]:
    from lime.lime_tabular import LimeTabularExplainer

    # Tiny synthetic background around the request for online latency
    rng = np.random.default_rng(42)
    base = row.to_numpy().reshape(-1)
    noise = rng.normal(0, 0.05, size=(200, len(feature_order)))
    background = np.clip(base + noise, 0, None)
    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_order,
        class_names=["no_diabetes", "diabetes_risk"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )

    def predict_fn(data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data, columns=feature_order)
        return model.predict_proba(df)

    exp = explainer.explain_instance(
        data_row=base,
        predict_fn=predict_fn,
        num_features=min(10, len(feature_order)),
        num_samples=300,
    )
    mapped = {c: 0.0 for c in feature_order}
    for label, weight in exp.as_list(label=1):
        for c in feature_order:
            if label.startswith(c) or label == c:
                mapped[c] += float(weight)
                break
    return mapped


app = FastAPI(
    title="Explainable Diabetes Risk API",
    description="Screening aid API — not a diagnostic service.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        art = load_production_artifact()
        available = True
        name = art.get("definition_name")
        threshold = art.get("threshold", 0.5)
    except FileNotFoundError:
        available = False
        name = None
        threshold = None
    schema = load_schema()
    return {
        "available": available,
        "model_name": name,
        "threshold": threshold,
        "feature_order": schema.feature_order,
        "intended_use": "preliminary diabetes risk screening aid",
        "out_of_scope": "diagnosis, treatment decisions, emergency triage",
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        art = load_production_artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    model = art.get("calibrated_model") or art["model"]
    feature_order = art["feature_order"]
    threshold = float(art.get("threshold", 0.5))
    payload = req.features.model_dump()
    row = pd.DataFrame([[payload[c] for c in feature_order]], columns=feature_order)

    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    label = int(proba >= threshold)
    explanations = None
    if req.include_explanations:
        t0 = time.perf_counter()
        ablation = _ablation_contributions(model, row, proba)
        lime_scores: dict[str, float] = {}
        lime_error = None
        try:
            lime_scores = _lime_contributions(model, row, feature_order)
        except Exception as exc:  # noqa: BLE001
            lime_error = str(exc)
        elapsed = time.perf_counter() - t0
        agree = agreement_metrics(ablation, lime_scores, top_k=3) if lime_scores else None
        # For API we expose ablation as a fast surrogate and LIME when available.
        # Full offline SHAP remains in src.explainability.generate.
        safe = render_safe_explanations(ablation if req.audience == "patient" else ablation, top_k=5)
        explanations = {
            "status": "ok_partial_xai",
            "method_notes": {
                "ablation": "Fast local Δ-probability vs reference values — association only, not SHAP.",
                "lime": "Optional online LIME (stochastic). Offline SHAP+LIME preferred for research.",
                "agreement": (
                    "If present, ablation-vs-LIME concordance is NOT predictive confidence."
                ),
            },
            "audience": req.audience,
            "ablation_top": render_safe_explanations(ablation, top_k=5),
            "lime_top": render_safe_explanations(lime_scores, top_k=5) if lime_scores else None,
            "lime_error": lime_error,
            "agreement": agree,
            "safe_explanations": safe,
            "latency_seconds": float(elapsed),
            "safe_note": (
                "Do not interpret feature contributions as causal. "
                "Clinical review of templates is pending. Screening aid only."
            ),
        }

    return PredictResponse(
        risk_probability=proba,
        risk_percent=round(proba * 100.0, 2),
        predicted_label=label,
        threshold=threshold,
        model_name=art.get("definition_name"),
        model_version=str(art.get("model_version", "phase1-dev")),
        disclaimer=DISCLAIMER,
        explanations=explanations,
    )
