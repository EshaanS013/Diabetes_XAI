"""Aggregate Phase-1 metrics from one or more experiment directories into results/phase1/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT, ensure_dir, save_json, setup_logging

logger = setup_logging()


def _load_model_metrics(model_dir: Path) -> dict | None:
    path = model_dir / "metrics.json"
    if not path.exists():
        path = model_dir / "metrics_partial.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(experiment_dirs: list[Path], out_dir: Path) -> Path:
    rows = []
    completed = []
    failures = []
    for exp in experiment_dirs:
        models_root = exp / "models"
        if not models_root.exists():
            continue
        for model_dir in sorted(models_root.iterdir()):
            if not model_dir.is_dir():
                continue
            payload = _load_model_metrics(model_dir)
            if payload is None:
                continue
            if payload.get("status") == "failed":
                failures.append(payload)
                continue
            tm = payload.get("test_metrics") or payload.get("validation_metrics") or {}
            row = {
                "model": payload.get("model", model_dir.name),
                "display_name": payload.get("display_name"),
                "status": payload.get("status", "completed"),
                "accuracy": tm.get("accuracy"),
                "precision": tm.get("precision"),
                "recall": tm.get("recall"),
                "specificity": tm.get("specificity"),
                "f1": tm.get("f1"),
                "roc_auc": tm.get("roc_auc"),
                "pr_auc": tm.get("pr_auc"),
                "brier_score": tm.get("brier_score"),
                "ece": tm.get("ece"),
                "cv_best_score": (payload.get("tuning") or {}).get("cv_best_score"),
                "training_time": payload.get("training_seconds"),
                "inference_ms_per_row": tm.get("inference_ms_per_row"),
                "decision_threshold": payload.get("decision_threshold"),
                "random_seed": payload.get("random_seed"),
                "calibrated": payload.get("calibrated"),
                "metrics_split": "test" if payload.get("test_metrics") else "validation",
                "source_experiment": exp.name,
            }
            # Prefer later experiment if same model appears twice
            rows = [r for r in rows if r["model"] != row["model"]]
            rows.append(row)
            completed.append(payload)

    ensure_dir(out_dir)
    df = pd.DataFrame(rows).sort_values("model")
    csv_path = out_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    save_json(rows, out_dir / "model_comparison.json")
    save_json(
        {
            "completed": completed,
            "failures": failures,
            "source_experiments": [str(p) for p in experiment_dirs],
            "official": True,
        },
        out_dir / "test_metrics.json",
    )
    logger.info("Wrote %s (%s models)", csv_path, len(rows))
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment directory names or paths under artifacts/experiments",
    )
    args = parser.parse_args()
    root = PROJECT_ROOT / "artifacts" / "experiments"
    dirs = []
    for item in args.experiments:
        p = Path(item)
        if not p.is_absolute():
            p = root / item if not p.exists() else p
        dirs.append(p)
    aggregate(dirs, PROJECT_ROOT / "results" / "phase1")


if __name__ == "__main__":
    main()
