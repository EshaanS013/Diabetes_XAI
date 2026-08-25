from __future__ import annotations

import copy
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.calibration.calibrate import fit_calibrator
from src.data.load import load_raw_dataframe, xy_from_dataframe
from src.data.schema import load_schema
from src.data.split import load_splits
from src.evaluation.metrics import bootstrap_metric_cis, classification_metrics
from src.models.registry import ModelDefinition, get_model_definition
from src.preprocessing.pipeline import build_model_pipeline
from src.utils.config import (
    PROJECT_ROOT,
    ensure_dir,
    environment_metadata,
    load_base_config,
    load_yaml,
    save_json,
    set_global_seed,
    setup_logging,
)

logger = setup_logging()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert Optuna-friendly encodings back to estimator parameter types."""
    out = dict(params)
    key = "model__hidden_layer_sizes"
    if key in out and isinstance(out[key], str):
        out[key] = tuple(int(x) for x in out[key].split("_"))
    return out


def _suggest_params(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, spec in space.items():
        if isinstance(spec, list):
            params[key] = trial.suggest_categorical(key, spec)
        elif isinstance(spec, tuple):
            kind = spec[0]
            if kind == "int":
                params[key] = trial.suggest_int(key, int(spec[1]), int(spec[2]))
            elif kind == "float":
                log = len(spec) > 3 and spec[3] == "log"
                params[key] = trial.suggest_float(key, float(spec[1]), float(spec[2]), log=log)
            elif kind == "categorical":
                params[key] = trial.suggest_categorical(key, list(spec[1]))
            else:
                raise ValueError(f"Unknown search space kind: {kind}")
        else:
            raise ValueError(f"Invalid search space for {key}: {spec}")
    return _normalize_params(params)


def tune_model(
    definition: ModelDefinition,
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    primary_metric: str,
    cv_folds: int,
    seed: int,
    n_trials: int,
) -> tuple[Any, dict[str, Any]]:
    scoring = primary_metric
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    if definition.tuning_strategy == "grid_small" and definition.search_space:
        grid = {
            k: (v if isinstance(v, list) else list(v[1]) if isinstance(v, tuple) and v[0] == "categorical" else v)
            for k, v in definition.search_space.items()
        }
        # Only keep list-valued keys for GridSearchCV
        grid = {k: v for k, v in grid.items() if isinstance(v, list)}
        if not grid:
            grid = {}
        if grid:
            search = GridSearchCV(
                pipeline,
                param_grid=grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            info = {
                "strategy": "grid_small",
                "best_params": search.best_params_,
                "cv_best_score": float(search.best_score_),
                "cv_results_summary": {
                    "mean_test_score": [float(x) for x in search.cv_results_["mean_test_score"]],
                    "std_test_score": [float(x) for x in search.cv_results_["std_test_score"]],
                },
            }
            return best, info

    if definition.tuning_strategy == "optuna" and definition.search_space:
        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial, definition.search_space)
            pipe = copy.deepcopy(pipeline)
            pipe.set_params(**params)
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )
            return float(np.mean(scores["test_score"]))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = _normalize_params(study.best_params)
        best_pipe = copy.deepcopy(pipeline)
        best_pipe.set_params(**best_params)
        best_pipe.fit(X_train, y_train)
        info = {
            "strategy": "optuna",
            "best_params": best_params,
            "cv_best_score": float(study.best_value),
            "n_trials": n_trials,
        }
        return best_pipe, info

    # No tuning — fit defaults and report CV stability
    cv_res = cross_validate(
        copy.deepcopy(pipeline),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    pipeline.fit(X_train, y_train)
    info = {
        "strategy": "default_params",
        "best_params": {},
        "cv_best_score": float(np.mean(cv_res["test_score"])),
        "cv_mean": float(np.mean(cv_res["test_score"])),
        "cv_std": float(np.std(cv_res["test_score"])),
        "cv_fold_scores": [float(x) for x in cv_res["test_score"]],
    }
    return pipeline, info


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # squash to (0,1) for metric compatibility — not calibrated
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(X)
    return preds.astype(float)


def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    base_cfg: dict[str, Any],
    phase1_cfg: dict[str, Any],
    experiment_dir: Path,
    evaluate_test: bool,
) -> dict[str, Any]:
    definition = get_model_definition(model_name)
    seed = int(base_cfg["reproducibility"]["random_seed"])
    pre_cfg = base_cfg["preprocessing"]
    primary_metric = phase1_cfg.get("primary_metric", base_cfg["training"]["primary_metric"])
    cv_folds = int(phase1_cfg.get("cv_folds", base_cfg["training"]["cv_folds"]))

    estimator = definition.estimator_factory(seed)
    pipeline = build_model_pipeline(
        estimator,
        scale=definition.requires_scaling,
        use_smote=bool(pre_cfg.get("use_smote", True)),
        smote_random_state=int(pre_cfg.get("smote_random_state", seed)),
        smote_k_neighbors=int(pre_cfg.get("smote_k_neighbors", 5)),
        bmi_iqr_cap=bool(pre_cfg.get("bmi_iqr_cap", True)),
        bmi_iqr_multiplier=float(pre_cfg.get("bmi_iqr_multiplier", 1.5)),
    )

    n_trials = (
        int(base_cfg["training"]["tuning_trials_heavy"])
        if definition.tuning_strategy == "optuna"
        else int(base_cfg["training"]["tuning_trials_simple"])
    )
    # Smoke / subsampled development runs: keep tuning cheap
    if phase1_cfg.get("dev_train_subsample"):
        n_trials = min(n_trials, 5)
        if definition.tuning_strategy == "optuna":
            n_trials = min(n_trials, 3)

    t0 = time.perf_counter()
    model, tune_info = tune_model(
        definition,
        pipeline,
        X_train,
        y_train,
        primary_metric=primary_metric,
        cv_folds=cv_folds,
        seed=seed,
        n_trials=n_trials,
    )
    train_seconds = time.perf_counter() - t0

    # Optional calibration on validation only (for probability quality / % risk display).
    # Primary classification comparison uses the uncalibrated model — isotonic on an
    # imbalanced validation set can crush recall at the default 0.5 threshold.
    calibrated = None
    if base_cfg["evaluation"].get("calibrate", True):
        try:
            calibrated = fit_calibrator(
                model,
                X_val,
                y_val.to_numpy(),
                method=base_cfg["evaluation"].get("calibration_method", "isotonic"),
            )
        except Exception as exc:
            logger.warning("Calibration failed for %s: %s", model_name, exc)

    default_threshold = float(base_cfg["evaluation"]["default_threshold"])

    def _eval(
        split_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        estimator: Any,
        threshold: float,
    ) -> dict[str, Any]:
        t_inf = time.perf_counter()
        proba = _predict_proba(estimator, X)
        inf_s = time.perf_counter() - t_inf
        metrics = classification_metrics(y.to_numpy(), proba, threshold=threshold)
        metrics["inference_seconds_total"] = float(inf_s)
        metrics["inference_ms_per_row"] = float(1000.0 * inf_s / max(len(y), 1))
        metrics["split"] = split_name
        return metrics, proba

    # Threshold tuning on validation (uncalibrated) — never on test
    val_metrics_default, val_proba = _eval(
        "validation", X_val, y_val, estimator=model, threshold=default_threshold
    )
    best_threshold = default_threshold
    best_val_f1 = float(val_metrics_default["f1"])
    for t in np.linspace(0.1, 0.9, 81):
        m = classification_metrics(y_val.to_numpy(), val_proba, threshold=float(t))
        if m["f1"] > best_val_f1:
            best_val_f1 = float(m["f1"])
            best_threshold = float(t)

    val_metrics, _ = _eval(
        "validation", X_val, y_val, estimator=model, threshold=best_threshold
    )
    calibrated_val = None
    if calibrated is not None:
        calibrated_val, _ = _eval(
            "validation_calibrated",
            X_val,
            y_val,
            estimator=calibrated,
            threshold=default_threshold,
        )
        calibrated_val = {
            k: v for k, v in calibrated_val.items() if k not in {"roc_curve", "pr_curve"}
        }

    threshold = best_threshold
    result: dict[str, Any] = {
        "model": model_name,
        "display_name": definition.display_name,
        "family": definition.family,
        "status": "completed",
        "training_seconds": float(train_seconds),
        "tuning": tune_info,
        "requires_scaling": definition.requires_scaling,
        "explainability_strategy": definition.explainability_strategy,
        "decision_threshold": threshold,
        "default_threshold": default_threshold,
        "threshold_tuned_on": "validation_uncalibrated_f1",
        "random_seed": seed,
        "primary_metric": primary_metric,
        "validation_metrics": {
            k: v for k, v in val_metrics.items() if k not in {"roc_curve", "pr_curve"}
        },
        "validation_metrics_default_threshold": {
            k: v
            for k, v in val_metrics_default.items()
            if k not in {"roc_curve", "pr_curve"}
        },
        "validation_metrics_calibrated_default_threshold": calibrated_val,
        "calibrated": calibrated is not None,
        "error": None,
    }

    model_dir = ensure_dir(experiment_dir / "models" / model_name)
    joblib.dump(
        {
            "model": model,
            "calibrated_model": calibrated,
            "definition_name": model_name,
            "feature_order": list(X_train.columns),
            "threshold": threshold,
            "selection_status": phase1_cfg.get("selection_status"),
        },
        model_dir / "artifact.joblib",
    )
    save_json(result, model_dir / "metrics_partial.json")
    save_json(tune_info, model_dir / "hyperparameters.json")

    if evaluate_test:
        test_metrics, test_proba = _eval(
            "test", X_test, y_test, estimator=model, threshold=threshold
        )
        cis = bootstrap_metric_cis(
            y_test.to_numpy(),
            test_proba,
            threshold=threshold,
            n_bootstrap=int(base_cfg["evaluation"].get("bootstrap_ci_samples", 200)),
            seed=seed,
        )
        result["test_metrics"] = {
            k: v for k, v in test_metrics.items() if k not in {"roc_curve", "pr_curve"}
        }
        result["test_metric_cis"] = cis
        save_json(
            {
                "roc_curve": test_metrics["roc_curve"],
                "pr_curve": test_metrics["pr_curve"],
                "confusion_matrix": test_metrics["confusion_matrix"],
            },
            model_dir / "test_curves.json",
        )
        save_json(result, model_dir / "metrics.json")
    return result


def run_phase1(
    *,
    config_path: str = "configs/phase1_models.yaml",
    force_official: bool = False,
    evaluate_test: bool = True,
    models_override: list[str] | None = None,
) -> dict[str, Any]:
    base_cfg = load_base_config()
    phase1_cfg = load_yaml(config_path)
    seed = int(base_cfg["reproducibility"]["random_seed"])
    set_global_seed(seed)

    selection_status = phase1_cfg.get("selection_status", "unknown")
    allow = bool(phase1_cfg.get("allow_official_phase1_run", False)) or force_official
    if selection_status != "confirmed" and not allow:
        logger.warning(
            "Phase-1 model list is '%s' (not confirmed). "
            "Running in DEVELOPMENT mode. Official claims require confirmation "
            "or --force-official. See docs/source_of_truth.md.",
            selection_status,
        )

    models = models_override or list(phase1_cfg["phase1_models"])
    for name in models:
        get_model_definition(name)  # validate early

    schema = load_schema()
    df = load_raw_dataframe(base_cfg, schema)
    X, y = xy_from_dataframe(df, schema)
    splits = load_splits()

    X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
    X_val, y_val = X.iloc[splits["validation"]], y.iloc[splits["validation"]]
    X_test, y_test = X.iloc[splits["test"]], y.iloc[splits["test"]]

    subsample = phase1_cfg.get("dev_train_subsample")
    if subsample:
        n = int(subsample)
        logger.warning("DEV stratified subsample enabled: using %s training rows", n)
        from sklearn.model_selection import train_test_split

        if n < len(X_train):
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=n,
                stratify=y_train,
                random_state=seed,
            )

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    experiment_dir = ensure_dir(PROJECT_ROOT / base_cfg["paths"]["experiments_dir"] / experiment_id)
    results_dir = ensure_dir(PROJECT_ROOT / base_cfg["paths"]["results_dir"])

    meta = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_status": selection_status,
        "official_run": bool(allow and selection_status == "confirmed") or force_official,
        "models": models,
        "primary_metric": phase1_cfg.get("primary_metric"),
        "environment": environment_metadata(),
        "base_config": base_cfg,
        "phase1_config": phase1_cfg,
        "disclaimer": (
            "Not a diagnostic device. Results are for research screening evaluation only. "
            "Do not fabricate or invent metrics — only measured values appear below."
        ),
    }
    save_json(meta, experiment_dir / "metadata.json")

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for name in models:
        logger.info("=== Training model: %s ===", name)
        try:
            result = train_single_model(
                name,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                base_cfg=base_cfg,
                phase1_cfg=phase1_cfg,
                experiment_dir=experiment_dir,
                evaluate_test=evaluate_test,
            )
            completed.append(result)
            logger.info(
                "Completed %s | val_f1=%.4f val_recall=%.4f",
                name,
                result["validation_metrics"]["f1"],
                result["validation_metrics"]["recall"],
            )
        except Exception as exc:
            err = {
                "model": name,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(err)
            save_json(err, experiment_dir / "models" / name / "failure.json")
            logger.error("Model %s failed: %s", name, exc)

    comparison_rows = []
    for r in completed:
        tm = r.get("test_metrics") or r.get("validation_metrics") or {}
        row = {
            "model": r["model"],
            "display_name": r["display_name"],
            "status": r["status"],
            "accuracy": tm.get("accuracy"),
            "precision": tm.get("precision"),
            "recall": tm.get("recall"),
            "specificity": tm.get("specificity"),
            "f1": tm.get("f1"),
            "roc_auc": tm.get("roc_auc"),
            "pr_auc": tm.get("pr_auc"),
            "brier_score": tm.get("brier_score"),
            "ece": tm.get("ece"),
            "cv_best_score": (r.get("tuning") or {}).get("cv_best_score"),
            "training_time": r.get("training_seconds"),
            "inference_ms_per_row": tm.get("inference_ms_per_row"),
            "decision_threshold": r.get("decision_threshold"),
            "random_seed": r.get("random_seed"),
            "calibrated": r.get("calibrated"),
            "metrics_split": "test" if r.get("test_metrics") else "validation",
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = results_dir / "model_comparison.csv"
    comparison_json = results_dir / "model_comparison.json"
    comparison_df.to_csv(comparison_csv, index=False)
    save_json(comparison_rows, comparison_json)
    save_json(
        {"completed": completed, "failures": failures, "experiment_id": experiment_id},
        results_dir / "test_metrics.json",
    )
    save_json(
        {"completed": [r["model"] for r in completed], "failed": [f["model"] for f in failures]},
        experiment_dir / "summary.json",
    )

    # Also copy comparison into experiment dir
    comparison_df.to_csv(experiment_dir / "model_comparison.csv", index=False)

    summary = {
        "experiment_id": experiment_id,
        "completed": [r["model"] for r in completed],
        "failed": [f["model"] for f in failures],
        "results_dir": str(results_dir),
        "experiment_dir": str(experiment_dir),
        "selection_status": selection_status,
        "official_run": meta["official_run"],
    }
    logger.info("Phase-1 run finished: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Phase-1 diabetes risk models")
    parser.add_argument("--config", default="configs/phase1_models.yaml")
    parser.add_argument(
        "--force-official",
        action="store_true",
        help="Override pending-confirmation lock (use only with explicit human approval)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model name overrides (must exist in registry)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Evaluate on validation only (development)",
    )
    args = parser.parse_args(argv)
    run_phase1(
        config_path=args.config,
        force_official=args.force_official,
        evaluate_test=not args.skip_test,
        models_override=args.models,
    )


if __name__ == "__main__":
    main()
