"""Compact tracked curve arrays produced by runs from before compaction was enabled."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.metrics import compact_curve
from src.utils.config import PROJECT_ROOT, load_json, save_json


def compact_file(path: Path) -> None:
    payload = load_json(path)
    for entry in payload.get("completed", []):
        curves = entry.get("test_curves") or {}
        for name in ("roc_curve", "pr_curve"):
            if name in curves:
                curves[name].pop("thresholds", None)
                curves[name] = compact_curve(curves[name])
    save_json(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    compact_file(path)


if __name__ == "__main__":
    main()
