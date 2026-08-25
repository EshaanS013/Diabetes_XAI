from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import PROJECT_ROOT, ensure_dir, load_base_config, save_json, setup_logging

logger = setup_logging()


def make_stratified_splits(
    y: pd.Series,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        stratify=y,
        random_state=random_seed,
    )
    relative_test = test_ratio / (validation_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=y.iloc[temp_idx],
        random_state=random_seed,
    )
    splits = {
        "train": np.sort(train_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, np.ndarray]) -> None:
    sets = {k: set(map(int, v)) for k, v in splits.items()}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for a, b in pairs:
        overlap = sets[a] & sets[b]
        if overlap:
            raise AssertionError(f"Split overlap between {a} and {b}: {len(overlap)} indices")


def save_splits(
    splits: dict[str, np.ndarray],
    metadata: dict[str, Any],
    out_dir: str | Path | None = None,
) -> Path:
    cfg = load_base_config()
    out_dir = Path(out_dir) if out_dir else PROJECT_ROOT / cfg["paths"]["splits_dir"]
    ensure_dir(out_dir)
    for name, idx in splits.items():
        pd.Series(idx, name="index").to_csv(out_dir / f"{name}_indices.csv", index=False)
    save_json(metadata, out_dir / "split_metadata.json")
    # Convenience mirror under data/splits
    mirror = PROJECT_ROOT / "data" / "splits"
    ensure_dir(mirror)
    for name, idx in splits.items():
        pd.Series(idx, name="index").to_csv(mirror / f"{name}_indices.csv", index=False)
    save_json(metadata, mirror / "split_metadata.json")
    logger.info("Saved split indices to %s", out_dir)
    return out_dir


def load_splits(splits_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    cfg = load_base_config()
    splits_dir = Path(splits_dir) if splits_dir else PROJECT_ROOT / cfg["paths"]["splits_dir"]
    out: dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        path = splits_dir / f"{name}_indices.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        out[name] = pd.read_csv(path)["index"].to_numpy()
    assert_disjoint(out)
    return out
