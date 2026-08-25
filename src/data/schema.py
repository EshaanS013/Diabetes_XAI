from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT, load_yaml


@dataclass(frozen=True)
class DataSchema:
    target: str
    feature_order: list[str]
    features: dict[str, dict[str, Any]]
    allowed_target_values: list[int]
    scale_sensitive_features: list[str]
    raw: dict[str, Any]

    @property
    def n_features(self) -> int:
        return len(self.feature_order)

    def required_columns(self) -> list[str]:
        return [self.target, *self.feature_order]


def load_schema(path: str | Path = "configs/data_schema.yaml") -> DataSchema:
    raw = load_yaml(path)
    target = raw["target"]["name"]
    feature_order = list(raw["feature_order"])
    features = dict(raw["features"])
    allowed = [int(v) for v in raw["target"]["allowed_values"]]
    scale_sensitive = list(raw.get("scale_sensitive_features", []))
    # Consistency checks
    missing = [f for f in feature_order if f not in features]
    if missing:
        raise ValueError(f"feature_order entries missing from features: {missing}")
    extras = [f for f in features if f not in feature_order]
    if extras:
        raise ValueError(f"features not listed in feature_order: {extras}")
    return DataSchema(
        target=target,
        feature_order=feature_order,
        features=features,
        allowed_target_values=allowed,
        scale_sensitive_features=scale_sensitive,
        raw=raw,
    )


def schema_path() -> Path:
    return PROJECT_ROOT / "configs" / "data_schema.yaml"
