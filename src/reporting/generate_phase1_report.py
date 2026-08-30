from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT, ensure_dir, load_json, load_yaml, setup_logging

logger = setup_logging()


def generate_phase1_report() -> Path:
    results = PROJECT_ROOT / "results" / "phase1"
    reports = ensure_dir(PROJECT_ROOT / "reports")
    phase1 = load_yaml("configs/phase1_models.yaml")
    comparison_path = results / "model_comparison.csv"

    lines = [
        "# Phase 1 Experiment Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Model selection status:** `{phase1.get('selection_status')}`",
        "",
        "## Important",
        "",
        "- This system is a **screening aid**, not a diagnostic device.",
        "- Metrics below are included **only if measured**. Missing values are marked TBD.",
        "- False negatives matter in healthcare screening; do not select on accuracy alone.",
        "- SHAP/LIME agreement is explanation concordance, **not** predictive confidence.",
        "",
        "## Configured Phase-1 models",
        "",
    ]
    for m in phase1.get("phase1_models", []):
        lines.append(f"- `{m}`")

    lines += [
        "",
        f"**Rationale (proposed):** {phase1.get('selection_rationale', '').strip()}",
        "",
        "## Results",
        "",
    ]

    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        lines.append("```")
        lines.append(df.to_string(index=False))
        lines.append("```")
        lines += ["", "### Model selection", ""]

        if "f1" in df.columns and df["f1"].notna().any():
            best_f1 = df.loc[df["f1"].idxmax()]
            best_recall = df.loc[df["recall"].idxmax()] if "recall" in df.columns else best_f1
            best_auc = df.loc[df["roc_auc"].idxmax()] if "roc_auc" in df.columns else best_f1
            selected = best_f1["model"]
            lines += [
                f"**Selected model:** `{selected}`",
                "",
                "**Justification:**",
                f"- Highest F1 on held-out test ({best_f1['f1']:.4f})",
                f"- Competitive recall ({best_f1.get('recall', float('nan')):.4f}); "
                f"highest recall: `{best_recall['model']}` ({best_recall['recall']:.4f})",
                f"- ROC-AUC: {best_f1.get('roc_auc', float('nan')):.4f} "
                f"(leader: `{best_auc['model']}` {best_auc['roc_auc']:.4f})",
                "- Do **not** select on accuracy alone; false negatives matter in screening.",
                "",
            ]
        else:
            lines += [
                "**Tentative selected model:** TBD - generated after experimental run",
                "",
                "**Justification:** TBD - generated after experimental run",
                "",
            ]
    else:
        lines += [
            "TBD - generated after experimental run",
            "",
            "Run:",
            "",
            "```bash",
            "python -m src.data.prepare",
            "python -m src.training.train --config configs/phase1_models.yaml",
            "python -m src.reporting.generate_phase1_report",
            "```",
            "",
        ]

    lines += [
        "## Failures",
        "",
    ]
    metrics_path = results / "test_metrics.json"
    if metrics_path.exists():
        payload = load_json(metrics_path)
        failures = payload.get("failures", [])
        if failures:
            for f in failures:
                lines.append(f"- `{f.get('model')}`: {f.get('error')}")
        else:
            lines.append("No recorded failures in latest `test_metrics.json`.")
    else:
        lines.append("TBD - generated after experimental run")

    lines += [
        "",
        "## Ethics & limitations (must appear in paper/presentation)",
        "",
        "- False reassurance risk from false negatives",
        "- Misleading explanations / disagreement between SHAP and LIME",
        "- SHAP/LIME are not causal",
        "- US survey population may not transfer to Indian clinical use without local validation",
        "- Probability displays require calibration checks",
        "",
    ]

    out = reports / "phase1_experiment_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # also mirror under results
    (results / "experiment_summary.md").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    generate_phase1_report()


if __name__ == "__main__":
    main()
