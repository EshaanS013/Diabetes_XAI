from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data.schema import DataSchema, load_schema
from src.utils.config import PROJECT_ROOT, load_base_config, setup_logging

logger = setup_logging()


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_raw_csv(cfg: dict | None = None) -> Path:
    cfg = cfg or load_base_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    filename = cfg["data"]["raw_filename"]
    path = raw_dir / filename
    if not path.exists():
        # Also accept any single CSV in raw/
        csvs = sorted(raw_dir.glob("*.csv"))
        if len(csvs) == 1:
            logger.warning("Configured filename missing; using sole CSV found: %s", csvs[0].name)
            return csvs[0]
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place the BRFSS2015 binary CSV in data/raw/ or run: "
            "python scripts/download_dataset.py"
        )
    return path


def load_raw_dataframe(cfg: dict | None = None, schema: DataSchema | None = None) -> pd.DataFrame:
    cfg = cfg or load_base_config()
    schema = schema or load_schema()
    path = resolve_raw_csv(cfg)
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded shape=%s hash=%s", df.shape, file_sha256(path))
    return df


def xy_from_dataframe(df: pd.DataFrame, schema: DataSchema | None = None) -> tuple[pd.DataFrame, pd.Series]:
    schema = schema or load_schema()
    missing = [c for c in schema.required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[schema.feature_order].copy()
    y = df[schema.target].astype(int).copy()
    return X, y
