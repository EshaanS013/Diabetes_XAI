from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.load import load_raw_dataframe, xy_from_dataframe
from src.data.schema import load_schema
from src.data.split import load_splits
from src.utils.config import PROJECT_ROOT, ensure_dir, load_base_config, load_yaml, save_json, set_global_seed, setup_logging

logger = setup_logging()


def _unwrap_estimator(artifact: dict[str, Any]) -> Any:
    model = artifact.get("calibrated_model") or artifact["model"]
    return model


def _get_feature_names(artifact: dict[str, Any]) -> list[str]:
    return list(artifact["feature_order"])


def compute_shap_local(
    model: Any,
    X_row: pd.DataFrame,
    X_background: pd.DataFrame,
    *,
    strategy: str,
) -> dict[str, float]:
    import shap

    # Prefer TreeExplainer when possible
    explainer = None
    if strategy == "tree":
        try:
            # Walk to underlying tree model if pipeline
            est = model
            if hasattr(est, "named_steps") and "model" in est.named_steps:
                # calibrated wrappers complicate TreeSHAP — fall back to Kernel/Permutation
                raise RuntimeError("pipeline+tree: use permutation explainer")
            explainer = shap.TreeExplainer(est)
            values = explainer.shap_values(X_row)
            if isinstance(values, list):
                values = values[1]
            vals = np.asarray(values).reshape(-1)
            return {c: float(v) for c, v in zip(X_row.columns, vals)}
        except Exception as exc:
            logger.info("TreeSHAP unavailable (%s); using permutation explainer", exc)

    # Model-agnostic fallback (works with pipelines)
    bg = shap.sample(X_background, min(100, len(X_background)), random_state=42)
    explainer = shap.Explainer(model.predict_proba, bg)
    explanation = explainer(X_row)
    vals = np.asarray(explanation.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    vals = vals.reshape(-1)
    return {c: float(v) for c, v in zip(X_row.columns, vals)}


def compute_lime_local(
    model: Any,
    X_row: pd.DataFrame,
    X_train: pd.DataFrame,
    *,
    num_samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=list(X_train.columns),
        class_names=["no_diabetes", "diabetes_risk"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )

    def predict_fn(data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data, columns=X_train.columns)
        return model.predict_proba(df)

    exp = explainer.explain_instance(
        data_row=X_row.to_numpy().reshape(-1),
        predict_fn=predict_fn,
        num_features=len(X_train.columns),
        num_samples=num_samples,
    )
    # Map back to raw feature names (LIME may return binned labels)
    raw = dict(exp.as_list(label=1))
    # Best-effort: if label starts with feature name, attribute to that feature
    mapped: dict[str, float] = {c: 0.0 for c in X_train.columns}
    for label, weight in raw.items():
        matched = None
        for c in X_train.columns:
            if label.startswith(c):
                matched = c
                break
        if matched:
            mapped[matched] += float(weight)
    return mapped


def agreement_metrics(
    shap_scores: dict[str, float],
    lime_scores: dict[str, float],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    shap_top = [k for k, _ in sorted(shap_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]]
    lime_top = [k for k, _ in sorted(lime_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]]
    set_s, set_l = set(shap_top), set(lime_top)
    overlap = set_s & set_l
    jaccard = len(overlap) / len(set_s | set_l) if (set_s | set_l) else 0.0
    return {
        "shap_top_features": shap_top,
        "lime_top_features": lime_top,
        "top_k": top_k,
        "exact_top_k_set_match": set_s == set_l,
        "overlap_count": len(overlap),
        "jaccard_overlap": float(jaccard),
        "note": (
            "Explanation concordance only — NOT predictive confidence, certainty, or clinical truth."
        ),
    }


SAFE_TEMPLATES: dict[str, dict[str, str]] = {
    "BMI": {
        "positive": "Higher body-mass index contributed to the elevated risk score in this model.",
        "negative": "Lower body-mass index contributed toward a lower risk score in this model.",
        "modifiable": "true",
        "clinical_review_status": "pending",
    },
    "HighBP": {
        "positive": "A history of high blood pressure contributed to the elevated risk score.",
        "negative": "Absence of reported high blood pressure contributed toward a lower risk score.",
        "modifiable": "partially",
        "clinical_review_status": "pending",
    },
    "GenHlth": {
        "positive": "Poorer self-rated general health contributed to the elevated risk score.",
        "negative": "Better self-rated general health contributed toward a lower risk score.",
        "modifiable": "partially",
        "clinical_review_status": "pending",
    },
    "Age": {
        "positive": "Older age category contributed to the elevated risk score (non-modifiable).",
        "negative": "Younger age category contributed toward a lower risk score (non-modifiable).",
        "modifiable": "false",
        "clinical_review_status": "pending",
    },
    "PhysActivity": {
        "positive": "Low reported physical activity contributed to the elevated risk score.",
        "negative": "Reported physical activity contributed toward a lower risk score.",
        "modifiable": "true",
        "clinical_review_status": "pending",
    },
}


def render_safe_explanations(feature_scores: dict[str, float], top_k: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(feature_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    out = []
    for feat, score in ranked:
        direction = "positive" if score >= 0 else "negative"
        tmpl = SAFE_TEMPLATES.get(
            feat,
            {
                "positive": f"Feature '{feat}' increased the model risk score (association, not causation).",
                "negative": f"Feature '{feat}' decreased the model risk score (association, not causation).",
                "modifiable": "unknown",
                "clinical_review_status": "pending",
            },
        )
        out.append(
            {
                "feature": feat,
                "contribution": float(score),
                "direction": direction,
                "patient_safe_description": tmpl[direction]
                + " This reflects model association, not proven causation.",
                "modifiable": tmpl.get("modifiable", "unknown"),
                "clinical_review_status": tmpl.get("clinical_review_status", "pending"),
            }
        )
    return out


def explain_from_artifact(
    artifact_path: str | Path,
    *,
    n_samples: int = 20,
    strategy: str = "tree",
) -> dict[str, Any]:
    base_cfg = load_base_config()
    xai_cfg = base_cfg["explainability"]
    seed = int(base_cfg["reproducibility"]["random_seed"])
    set_global_seed(seed)

    artifact = joblib.load(artifact_path)
    model = _unwrap_estimator(artifact)
    feature_order = _get_feature_names(artifact)

    schema = load_schema()
    df = load_raw_dataframe(base_cfg, schema)
    X, y = xy_from_dataframe(df, schema)
    X = X[feature_order]
    splits = load_splits()
    X_train = X.iloc[splits["train"]]
    X_test = X.iloc[splits["test"]]

    bg_size = int(xai_cfg.get("shap_background_size", 200))
    X_bg = X_train.sample(n=min(bg_size, len(X_train)), random_state=seed)
    sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=seed)

    rows = []
    for idx, row in sample.iterrows():
        X_row = pd.DataFrame([row], columns=feature_order)
        t0 = time.perf_counter()
        try:
            shap_scores = compute_shap_local(model, X_row, X_bg, strategy=strategy)
            shap_s = time.perf_counter() - t0
        except Exception as exc:
            shap_scores = {}
            shap_s = time.perf_counter() - t0
            logger.warning("SHAP failed for idx=%s: %s", idx, exc)

        t1 = time.perf_counter()
        try:
            lime_scores = compute_lime_local(
                model,
                X_row,
                X_bg,
                num_samples=int(xai_cfg.get("lime_num_samples", 1000)),
                seed=seed,
            )
            lime_s = time.perf_counter() - t1
        except Exception as exc:
            lime_scores = {}
            lime_s = time.perf_counter() - t1
            logger.warning("LIME failed for idx=%s: %s", idx, exc)

        agree = agreement_metrics(
            shap_scores,
            lime_scores,
            top_k=int(xai_cfg.get("agreement_top_k", 3)),
        )
        rows.append(
            {
                "row_index": int(idx),
                "true_label": int(y.loc[idx]),
                "shap": shap_scores,
                "lime": lime_scores,
                "shap_seconds": float(shap_s),
                "lime_seconds": float(lime_s),
                "agreement": agree,
                "safe_explanations_shap": render_safe_explanations(shap_scores),
                "safe_explanations_lime": render_safe_explanations(lime_scores),
            }
        )

    # Aggregate concordance
    if rows:
        jaccards = [r["agreement"]["jaccard_overlap"] for r in rows]
        exact = [r["agreement"]["exact_top_k_set_match"] for r in rows]
        aggregate = {
            "n": len(rows),
            "mean_jaccard": float(np.mean(jaccards)),
            "exact_top_k_agreement_rate": float(np.mean(exact)),
            "interpretation": (
                "Measures SHAP/LIME explanation concordance only. "
                "Does NOT measure predictive confidence or clinical correctness."
            ),
        }
    else:
        aggregate = {"n": 0}

    out = {
        "artifact_path": str(artifact_path),
        "n_samples": len(rows),
        "aggregate_concordance": aggregate,
        "instances": rows,
        "ethics_reminder": (
            "SHAP and LIME are not causal inference. Agreement is not certainty. "
            "This system is a screening aid, not a diagnosis."
        ),
    }
    out_dir = ensure_dir(PROJECT_ROOT / "results" / "phase1" / "explainability")
    save_json(out, out_dir / "shap_lime_local.json")
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate SHAP/LIME explanations")
    parser.add_argument("--artifact", required=True, help="Path to model artifact.joblib")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--strategy", default="tree", choices=["tree", "linear", "kernel"])
    args = parser.parse_args(argv)
    explain_from_artifact(args.artifact, n_samples=args.n_samples, strategy=args.strategy)


if __name__ == "__main__":
    main()
