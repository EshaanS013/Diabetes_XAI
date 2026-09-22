"""Combine tracked Phase-1 and Phase-2 outputs without selecting on the test set."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from src.utils.config import PROJECT_ROOT, ensure_dir, load_json, save_json, setup_logging

logger = setup_logging()


def aggregate() -> None:
    sources = [PROJECT_ROOT / "results" / "phase1", PROJECT_ROOT / "results" / "phase2"]
    rows: list[dict] = []
    completed: list[dict] = []
    for source in sources:
        comparison_path = source / "model_comparison.json"
        metrics_path = source / "test_metrics.json"
        if not comparison_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Missing benchmark outputs in {source}")
        phase_rows = load_json(comparison_path)
        for row in phase_rows:
            row = dict(row)
            row["benchmark_phase"] = source.name
            rows.append(row)
        completed.extend(load_json(metrics_path).get("completed", []))

    if len({row["model"] for row in rows}) != 10:
        raise RuntimeError("Expected exactly ten unique models across Phase 1 and Phase 2")

    df = pd.DataFrame(rows)
    df["cv_rank"] = df["cv_best_score"].rank(method="min", ascending=False).astype(int)
    df["test_f1_rank_retrospective"] = df["f1"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values(["cv_rank", "model"])

    out_dir = ensure_dir(PROJECT_ROOT / "results" / "all_models")
    df.to_csv(out_dir / "model_comparison.csv", index=False)
    json_rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    save_json(json_rows, out_dir / "model_comparison.json")
    save_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selection_policy": (
                "Rank and any production-candidate recommendation use training-fold CV F1. "
                "Held-out test metrics are reported once for final retrospective evaluation, "
                "not used to choose the model."
            ),
            "recommended_by_cv": str(df.iloc[0]["model"]),
            "completed": completed,
        },
        out_dir / "test_metrics.json",
    )

    cols = [
        "cv_rank",
        "model",
        "benchmark_phase",
        "cv_best_score",
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "pr_auc",
        "calibrated_brier_score",
        "calibrated_ece",
    ]
    display = df[[c for c in cols if c in df.columns]].copy()
    headers = list(display.columns)
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for value in row:
            if isinstance(value, float) and math.isnan(value):
                values.append("not recorded")
            else:
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        table.append("| " + " | ".join(values) + " |")
    lines = [
        "# Ten-model benchmark",
        "",
        "Models are ranked by cross-validation F1. Test-set values are final evaluation only.",
        "",
        *table,
        "",
        "## Selection guardrail",
        "",
        (
            "The `test_f1_rank_retrospective` column must not be used for model selection. "
            "Clinical utility, subgroup analysis, calibration, latency, and XAI review remain "
            "required before promotion."
        ),
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote ten-model benchmark to %s", out_dir)


if __name__ == "__main__":
    aggregate()
