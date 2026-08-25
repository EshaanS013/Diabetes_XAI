#!/usr/bin/env python
"""Download CDC BRFSS2015 Diabetes Binary Health Indicators dataset into data/raw/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = OUT / "diabetes_binary_health_indicators_BRFSS2015.csv"


def main() -> None:
    if TARGET.exists() and TARGET.stat().st_size > 1_000_000:
        print(f"Dataset already present: {TARGET} ({TARGET.stat().st_size} bytes)")
        return

    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd
    except ImportError:
        print("Installing dependency tip: pip install ucimlrepo", file=sys.stderr)
        raise SystemExit(
            "ucimlrepo is required for automatic download. "
            "Install it or place the CSV manually in data/raw/."
        )

    print("Fetching UCI dataset id=891 via ucimlrepo...")
    ds = fetch_ucirepo(id=891)
    X = ds.data.features
    y = ds.data.targets
    df = pd.concat([y, X], axis=1)
    if "Diabetes_binary" not in df.columns:
        # Defensive rename if upstream changes
        target_col = y.columns[0]
        df = df.rename(columns={target_col: "Diabetes_binary"})
    df.to_csv(TARGET, index=False)
    print(f"Saved to {TARGET} ({TARGET.stat().st_size} bytes), shape={df.shape}")


if __name__ == "__main__":
    main()
