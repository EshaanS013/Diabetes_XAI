from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.load import file_sha256, load_raw_dataframe, resolve_raw_csv, xy_from_dataframe
from src.data.schema import load_schema
from src.data.split import make_stratified_splits, save_splits
from src.data.validate import audit_dataframe, write_data_quality_markdown
from src.utils.config import PROJECT_ROOT, ensure_dir, load_base_config, save_json, set_global_seed, setup_logging

logger = setup_logging()


def prepare_dataset(cfg: dict | None = None) -> dict:
    cfg = cfg or load_base_config()
    seed = int(cfg["reproducibility"]["random_seed"])
    set_global_seed(seed)
    schema = load_schema()

    raw_path = resolve_raw_csv(cfg)
    df = load_raw_dataframe(cfg, schema)
    report = audit_dataframe(df, schema, dataset_hash=file_sha256(raw_path))
    write_data_quality_markdown(report)
    if report.status != "ok":
        raise RuntimeError(f"Data validation failed: {report.errors}")

    X, y = xy_from_dataframe(df, schema)
    processed = pd.concat([y.rename(schema.target), X], axis=1)
    processed_dir = ensure_dir(PROJECT_ROOT / cfg["paths"]["processed_dir"])
    processed_path = processed_dir / "dataset.parquet"
    try:
        processed.to_parquet(processed_path, index=True)
    except Exception:
        processed_path = processed_dir / "dataset.csv"
        processed.to_csv(processed_path, index=True)
    logger.info("Wrote processed dataset to %s", processed_path)

    split_cfg = cfg["split"]
    splits = make_stratified_splits(
        y,
        train_ratio=float(split_cfg["train_ratio"]),
        validation_ratio=float(split_cfg["validation_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        random_seed=seed,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "split_version": split_cfg.get("split_version", "split-v1"),
        "dataset_version": cfg["data"].get("dataset_version"),
        "dataset_hash": report.dataset_hash,
        "n_rows": int(len(df)),
        "ratios": {
            "train": split_cfg["train_ratio"],
            "validation": split_cfg["validation_ratio"],
            "test": split_cfg["test_ratio"],
        },
        "sizes": {k: int(len(v)) for k, v in splits.items()},
        "target_prevalence": {
            k: {
                "positive": int(y.iloc[v].sum()),
                "negative": int((y.iloc[v] == 0).sum()),
                "prevalence": float(y.iloc[v].mean()),
            }
            for k, v in splits.items()
        },
        "note": "Test set is sacred — never use for fitting, tuning, SMOTE, or selection.",
    }
    save_splits(splits, metadata)

    summary = {
        "status": "prepared",
        "processed_path": str(processed_path),
        "dataset_hash": report.dataset_hash,
        "split_sizes": metadata["sizes"],
    }
    save_json(summary, PROJECT_ROOT / "reports" / "prepare_summary.json")
    logger.info("Dataset preparation complete: %s", summary)
    return summary


def main() -> None:
    prepare_dataset()


if __name__ == "__main__":
    main()
