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
        lines += [
            "",
            "### Model selection discussion (template)",
            "",
            "Compare models on recall, F1, ROC-AUC, precision, specificity, calibration,",
            "latency, and explainability compatibility. Do **not** auto-declare a winner",
            "from a single metric. Fill after reviewing measured results.",
            "",
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
