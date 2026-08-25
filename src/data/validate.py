from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import file_sha256, load_raw_dataframe, resolve_raw_csv
from src.data.schema import DataSchema, load_schema
from src.utils.config import PROJECT_ROOT, ensure_dir, load_base_config, save_json, setup_logging

logger = setup_logging()


@dataclass
class DataQualityReport:
    n_rows: int
    n_columns: int
    n_features: int
    target_counts: dict[str, int]
    target_proportions: dict[str, float]
    duplicate_rows: int
    missing_by_column: dict[str, int]
    total_missing: int
    invalid_target_rows: int
    range_violations: dict[str, int] = field(default_factory=dict)
    unexpected_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    dataset_hash: str | None = None
    status: str = "ok"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_ranges(df: pd.DataFrame, schema: DataSchema) -> dict[str, int]:
    violations: dict[str, int] = {}
    for name, meta in schema.features.items():
        if name not in df.columns:
            continue
        col = df[name]
        allowed = meta.get("allowed_values")
        if allowed is not None:
            bad = ~col.isin(allowed)
            # tolerate float encodings of ints (0.0 / 1.0)
            if bad.any():
                # try integer cast for near-integers
                as_int = pd.to_numeric(col, errors="coerce")
                bad = ~as_int.round().astype("Int64").isin(allowed)
            violations[name] = int(bad.sum())
            continue
        lo, hi = meta.get("min"), meta.get("max")
        if lo is not None or hi is not None:
            vals = pd.to_numeric(col, errors="coerce")
            mask = pd.Series(False, index=df.index)
            if lo is not None:
                mask |= vals < lo
            if hi is not None:
                mask |= vals > hi
            mask |= vals.isna() & col.notna()
            violations[name] = int(mask.sum())
    return violations


def audit_dataframe(df: pd.DataFrame, schema: DataSchema | None = None, dataset_hash: str | None = None) -> DataQualityReport:
    schema = schema or load_schema()
    errors: list[str] = []
    required = schema.required_columns()
    missing_cols = [c for c in required if c not in df.columns]
    unexpected = [c for c in df.columns if c not in required]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if schema.target in df.columns:
        y = pd.to_numeric(df[schema.target], errors="coerce")
        invalid_target = int((~y.isin(schema.allowed_target_values)).sum() + y.isna().sum())
        if invalid_target:
            errors.append(f"Invalid target rows: {invalid_target}")
        vc = y.value_counts(dropna=False).to_dict()
        target_counts = {str(k): int(v) for k, v in vc.items()}
        n = len(df) or 1
        target_props = {k: v / n for k, v in target_counts.items()}
    else:
        invalid_target = -1
        target_counts = {}
        target_props = {}

    missing_by_column = {c: int(df[c].isna().sum()) for c in df.columns}
    total_missing = int(sum(missing_by_column.values()))
    dupes = int(df.duplicated().sum())
    range_violations = _check_ranges(df, schema) if not missing_cols else {}
    hard_range = {k: v for k, v in range_violations.items() if v > 0}
    # Soft-warn on range violations (survey data can have rare coding quirks);
    # fail hard only on missing columns / invalid target.
    status = "ok" if not errors else "failed"
    return DataQualityReport(
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        n_features=schema.n_features,
        target_counts=target_counts,
        target_proportions=target_props,
        duplicate_rows=dupes,
        missing_by_column=missing_by_column,
        total_missing=total_missing,
        invalid_target_rows=invalid_target,
        range_violations=hard_range,
        unexpected_columns=unexpected,
        missing_columns=missing_cols,
        dataset_hash=dataset_hash,
        status=status,
        errors=errors,
    )


def write_data_quality_markdown(report: DataQualityReport, path: str | None = None) -> str:
    path = path or str(PROJECT_ROOT / "reports" / "data_quality_report.md")
    ensure_dir(PROJECT_ROOT / "reports")
    lines = [
        "# Data Quality Report",
        "",
        f"**Status:** {report.status}",
        f"**Rows:** {report.n_rows}",
        f"**Columns:** {report.n_columns}",
        f"**Features (schema):** {report.n_features}",
        f"**Dataset hash (SHA-256):** `{report.dataset_hash}`",
        "",
        "## Target distribution",
        "",
        "| Class | Count | Proportion |",
        "|---|---:|---:|",
    ]
    for k, v in report.target_counts.items():
        prop = report.target_proportions.get(k, 0.0)
        lines.append(f"| {k} | {v} | {prop:.4f} |")
    lines += [
        "",
        f"**Duplicate rows:** {report.duplicate_rows}",
        f"**Total missing values:** {report.total_missing}",
        "",
        "## Missing values by column",
        "",
        "| Column | Missing |",
        "|---|---:|",
    ]
    for c, m in report.missing_by_column.items():
        lines.append(f"| {c} | {m} |")
    lines += ["", "## Range / encoding violations", ""]
    if report.range_violations:
        lines += ["| Feature | Violations |", "|---|---:|"]
        for k, v in report.range_violations.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("None detected under schema rules.")
    if report.errors:
        lines += ["", "## Errors", ""]
        for e in report.errors:
            lines.append(f"- {e}")
    if report.unexpected_columns:
        lines += ["", "## Unexpected columns", "", ", ".join(report.unexpected_columns)]
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    save_json(report.to_dict(), PROJECT_ROOT / "reports" / "data_quality_report.json")
    return path


def main() -> None:
    cfg = load_base_config()
    schema = load_schema()
    path = resolve_raw_csv(cfg)
    df = load_raw_dataframe(cfg, schema)
    report = audit_dataframe(df, schema, dataset_hash=file_sha256(path))
    out = write_data_quality_markdown(report)
    logger.info("Data quality report written to %s (status=%s)", out, report.status)
    if report.status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
