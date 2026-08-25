"""Promote a trained experiment artifact to artifacts/production/model.joblib."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils.config import PROJECT_ROOT, setup_logging

logger = setup_logging()


def promote(artifact: Path, dest: Path | None = None) -> Path:
    dest = dest or (PROJECT_ROOT / "artifacts" / "production" / "model.joblib")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    shutil.copy2(artifact, dest)
    logger.info("Promoted %s -> %s", artifact, dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Path to artifact.joblib")
    args = parser.parse_args()
    promote(Path(args.artifact))


if __name__ == "__main__":
    main()
