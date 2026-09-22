from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from src.data.load import load_raw_dataframe
from src.data.schema import load_schema
from src.data.split import load_splits
from src.utils.config import PROJECT_ROOT, load_base_config, save_json, setup_logging

logger = setup_logging()


def audit_exact_duplicate_overlap(
    df: pd.DataFrame, splits: dict[str, Any]
) -> dict[str, Any]:
    """Quantify exact feature+target duplicate rows that cross frozen split boundaries."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).astype("uint64")
    split_hashes = {
        name: row_hashes.iloc[list(indices)].reset_index(drop=True)
        for name, indices in splits.items()
    }
    pairs: dict[str, Any] = {}
    for left, right in combinations(sorted(split_hashes), 2):
        left_values = split_hashes[left]
        right_values = split_hashes[right]
        shared = set(left_values.unique()).intersection(right_values.unique())
        pairs[f"{left}__{right}"] = {
            "shared_unique_row_hashes": len(shared),
            f"{left}_rows_affected": int(left_values.isin(shared).sum()),
            f"{right}_rows_affected": int(right_values.isin(shared).sum()),
        }

    duplicate_mask = row_hashes.duplicated(keep=False)
    return {
        "audit_scope": "exact feature-plus-target row duplicates",
        "total_rows": len(df),
        "duplicated_rows": int(duplicate_mask.sum()),
        "duplicate_excess_rows": int(len(df) - row_hashes.nunique()),
        "cross_split_pairs": pairs,
        "interpretation": (
            "Non-zero overlap can make a row-random test split optimistic. Split-v1 is retained "
            "for direct Phase-1 comparability; a duplicate-grouped split should be a separately "
            "versioned sensitivity analysis before clinical claims."
        ),
    }


def main() -> None:
    cfg = load_base_config()
    schema = load_schema()
    df = load_raw_dataframe(cfg, schema)
    report = audit_exact_duplicate_overlap(df, load_splits())
    out = PROJECT_ROOT / "results" / "phase2" / "duplicate_overlap_audit.json"
    save_json(report, out)
    logger.info("Wrote duplicate-overlap audit to %s", out)


if __name__ == "__main__":
    main()
