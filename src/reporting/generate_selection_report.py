"""Generate a model-selection discussion report from measured comparison CSV."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT, ensure_dir, load_yaml, setup_logging

logger = setup_logging()


def generate_selection_report(
    comparison_csv: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    comparison_csv = comparison_csv or (PROJECT_ROOT / "results" / "phase1" / "model_comparison.csv")
    out_path = out_path or (PROJECT_ROOT / "reports" / "model_selection_report.md")
    ensure_dir(out_path.parent)
    phase1 = load_yaml("configs/phase1_models.yaml")

    lines = [
        "# Model Selection Report (Phase 1)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Selection status:** `{phase1.get('selection_status')}`",
        "",
        "## Policy",
        "",
        "- Do **not** select on accuracy alone.",
        "- Prioritize recall/FN burden, F1, ROC-AUC, calibration, latency, explainability fit.",
        "- Screening aid only — not a diagnostic claim.",
        "",
    ]

    if not comparison_csv.exists():
        lines += ["## Results", "", "TBD - generated after experimental run", ""]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    df = pd.read_csv(comparison_csv)
    lines += ["## Measured comparison", "", "```", df.to_string(index=False), "```", ""]

    # Rank helpers on available columns (measured only)
    for col in ("recall", "f1", "roc_auc"):
        if col in df.columns and df[col].notna().any():
            best = df.loc[df[col].idxmax()]
            lines.append(
                f"- Highest **{col}**: `{best.get('model')}` "
                f"({col}={best[col]:.4f}) — not automatically the production choice."
            )

    if "f1" in df.columns and df["f1"].notna().any():
        selected = df.loc[df["f1"].idxmax()]
        lines += [
            "",
            "## Selected model",
            "",
            f"**{selected['model']}** — highest measured F1 ({selected['f1']:.4f}) and competitive "
            f"recall ({selected.get('recall', float('nan')):.4f}), ROC-AUC "
            f"({selected.get('roc_auc', float('nan')):.4f}).",
            "",
        ]
    else:
        lines += [
            "",
            "## Tentative recommendation",
            "",
            "**Selected model:** TBD - generated after experimental review",
            "",
        ]

    lines += [
        "**Justification checklist:**",
        "- [ ] Recall / false-negative burden acceptable for screening context",
        "- [ ] F1 competitive under imbalance",
        "- [ ] ROC-AUC / PR-AUC reviewed",
        "- [ ] Calibration (Brier/ECE) acceptable for % risk display",
        "- [ ] Inference latency acceptable for mobile",
        "- [ ] Explainability path (TreeSHAP vs kernel) feasible",
        "- [ ] Stability across CV folds reviewed",
        "",
        "## Caveats",
        "",
        f"- Official publication requires `selection_status: confirmed` (currently `{phase1.get('selection_status')}`).",
        "- Development/subsample runs must be labelled as such in the paper/panel.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return out_path


def main() -> None:
    generate_selection_report()


if __name__ == "__main__":
    main()
