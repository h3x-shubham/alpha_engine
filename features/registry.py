"""
Decorator-based feature registry.

Features self-register via the @feature decorator, declaring their name,
lookback period, and any dependencies. The registry is then used by the
FeatureEngine to determine computation order and required history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Any

import pandas as pd

logger = logging.getLogger("alpha_engine.features.registry")


@dataclass
class FeatureMeta:
    """Metadata for a registered feature."""
    name: str
    lookback: int
    category: str  # "technical", "microstructure", "fundamental"
    dependencies: list[str] = field(default_factory=list)
    compute_fn: Callable | None = None


class FeatureRegistry:
    """
    Central registry of all available features.

    Features register themselves via the @feature decorator.
    The engine queries this registry to know what to compute.
    """

    _instance: "FeatureRegistry | None" = None
    _registry: dict[str, FeatureMeta] = {}

    def __new__(cls) -> "FeatureRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._registry = {}
        return cls._instance

    def register(self, meta: FeatureMeta) -> None:
        """Register a feature."""
        self._registry[meta.name] = meta
        logger.debug("Registered feature: %s (lookback=%d)", meta.name, meta.lookback)

    def get(self, name: str) -> FeatureMeta:
        """Get metadata for a named feature."""
        if name not in self._registry:
            raise KeyError(f"Feature not registered: {name}")
        return self._registry[name]

    def get_by_category(self, category: str) -> list[FeatureMeta]:
        """Get all features in a category."""
        return [f for f in self._registry.values() if f.category == category]

    def get_enabled(self, feature_config: dict[str, Any]) -> list[FeatureMeta]:
        """
        Get features that are enabled in the feature config.

        Matches feature names from config against the registry.
        """
        enabled: list[FeatureMeta] = []
        for category, cat_cfg in feature_config.get("features", {}).items():
            if not cat_cfg.get("enabled", False):
                continue
            for indicator in cat_cfg.get("indicators", []):
                name = indicator["name"]
                if name in self._registry:
                    enabled.append(self._registry[name])
                else:
                    logger.warning("Feature '%s' in config but not registered", name)
        return enabled

    @property
    def all_features(self) -> dict[str, FeatureMeta]:
        """Return all registered features."""
        return dict(self._registry)

    @property
    def max_lookback(self) -> int:
        """Return the maximum lookback across all features."""
        if not self._registry:
            return 0
        return max(f.lookback for f in self._registry.values())


def feature(
    name: str,
    lookback: int,
    category: str = "technical",
    dependencies: list[str] | None = None,
) -> Callable:
    """
    Decorator to register a feature computation function.

    The decorated function should accept a per-ticker DataFrame (sorted
    by date) and return a Series of feature values.

    Example:
        @feature("momentum_5d", lookback=5)
        def momentum_5d(df: pd.DataFrame) -> pd.Series:
            return df["close"].pct_change(5)
    """
    def decorator(fn: Callable) -> Callable:
        meta = FeatureMeta(
            name=name,
            lookback=lookback,
            category=category,
            dependencies=dependencies or [],
            compute_fn=fn,
        )
        FeatureRegistry().register(meta)
        return fn
    return decorator
