from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT, ensure_dir, load_json, setup_logging

logger = setup_logging()


def plot_comparison_bars(comparison_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(comparison_csv)
    if df.empty:
        logger.warning("Empty comparison CSV — skipping plots")
        return
    metrics = ["recall", "f1", "roc_auc", "precision", "specificity"]
    present = [m for m in metrics if m in df.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    width = 0.15
    for i, m in enumerate(present):
        vals = df[m].tolist()
        ax.bar([xi + i * width for xi in x], vals, width=width, label=m)
    ax.set_xticks([xi + width * (len(present) - 1) / 2 for xi in x])
    ax.set_xticklabels(df["model"].tolist(), rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Phase-1 model comparison (measured values only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metric_bars.png", dpi=150)
    plt.close(fig)


def plot_confusion_from_test_metrics(test_metrics_path: Path, out_dir: Path) -> None:
    """Fallback: plot confusion matrices from results/phase1/test_metrics.json."""
    if not test_metrics_path.exists():
        return
    payload = load_json(test_metrics_path)
    cm_dir = ensure_dir(out_dir / "confusion_matrices")
    for entry in payload.get("completed", []):
        tm = entry.get("test_metrics") or {}
        cm = tm.get("confusion_matrix")
        if not cm:
            continue
        name = entry.get("model", "unknown")
        mat = np.array(
            [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]],
            dtype=float,
        )
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
        im = ax_cm.imshow(mat, cmap="Blues")
        ax_cm.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax_cm.set_yticks([0, 1], ["True 0", "True 1"])
        for (i, j), v in np.ndenumerate(mat):
            ax_cm.text(j, i, int(v), ha="center", va="center")
        ax_cm.set_title(f"Confusion — {name}")
        fig_cm.colorbar(im, ax=ax_cm, fraction=0.046)
        fig_cm.tight_layout()
        fig_cm.savefig(cm_dir / f"{name}.png", dpi=150)
        plt.close(fig_cm)
    logger.info("Plotted confusion matrices from %s", test_metrics_path)


def _latest_experiment_dir() -> Path | None:
    root = PROJECT_ROOT / "artifacts" / "experiments"
    exps = sorted(
        [p for p in root.glob("20*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return exps[0] if exps else None


def plot_curves_from_experiment(experiment_dir: Path, out_dir: Path) -> None:
    models_dir = experiment_dir / "models"
    if not models_dir.exists():
        return

    # ROC overlay
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    cm_dir = ensure_dir(out_dir / "confusion_matrices")

    any_curve = False
    for model_dir in sorted(models_dir.iterdir()):
        curves_path = model_dir / "test_curves.json"
        if not curves_path.exists():
            continue
        curves = load_json(curves_path)
        name = model_dir.name
        roc = curves.get("roc_curve", {})
        pr = curves.get("pr_curve", {})
        if roc.get("fpr") and roc.get("tpr"):
            ax_roc.plot(roc["fpr"], roc["tpr"], label=name)
            any_curve = True
        if pr.get("recall") and pr.get("precision"):
            ax_pr.plot(pr["recall"], pr["precision"], label=name)
        cm = curves.get("confusion_matrix", {})
        if cm:
            mat = np.array(
                [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]],
                dtype=float,
            )
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            im = ax_cm.imshow(mat, cmap="Blues")
            ax_cm.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            ax_cm.set_yticks([0, 1], ["True 0", "True 1"])
            for (i, j), v in np.ndenumerate(mat):
                ax_cm.text(j, i, int(v), ha="center", va="center")
            ax_cm.set_title(f"Confusion — {name}")
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046)
            fig_cm.tight_layout()
            fig_cm.savefig(cm_dir / f"{name}.png", dpi=150)
            plt.close(fig_cm)

    if any_curve:
        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("ROC curves (held-out test)")
        ax_roc.legend(fontsize=8)
        fig_roc.tight_layout()
        fig_roc.savefig(out_dir / "roc_curves.png", dpi=150)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision–Recall curves (held-out test)")
        ax_pr.legend(fontsize=8)
        fig_pr.tight_layout()
        fig_pr.savefig(out_dir / "precision_recall_curves.png", dpi=150)
    plt.close(fig_roc)
    plt.close(fig_pr)


def evaluate_main(experiment_dir: str | Path | None = None) -> None:
    results = PROJECT_ROOT / "results" / "phase1"
    comparison = results / "model_comparison.csv"
    if not comparison.exists():
        raise FileNotFoundError(
            f"Missing {comparison}. Run training first: python -m src.training.train"
        )
    out_dir = ensure_dir(results / "figures")
    plot_comparison_bars(comparison, out_dir)

    exp = Path(experiment_dir) if experiment_dir else _latest_experiment_dir()
    plotted_curves = False
    if exp and exp.exists():
        models_dir = exp / "models"
        if models_dir.exists() and any(models_dir.glob("*/test_curves.json")):
            plot_curves_from_experiment(exp, out_dir)
            plotted_curves = True
            logger.info("Used experiment curves from %s", exp)
    if not plotted_curves:
        logger.warning("No experiment curve artifacts — using test_metrics confusion fallback")
        plot_confusion_from_test_metrics(results / "test_metrics.json", out_dir)

    logger.info("Wrote evaluation figures to %s", out_dir)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default=None)
    args = parser.parse_args(argv)
    evaluate_main(args.experiment_dir)


if __name__ == "__main__":
    main()
