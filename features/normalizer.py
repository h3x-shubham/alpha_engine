"""
Cross-sectional normalization.

Normalizes features per-date across the universe so that each date's
cross-section has consistent scale. This is critical: raw features are
time-series values, but the model needs to score stocks *relative to
each other* on a given day.

Normalization is applied AFTER feature computation and BEFORE model input.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.types import Cols, PanelData, validate_panel

logger = logging.getLogger("alpha_engine.features.normalizer")


class CrossSectionalNormalizer:
    """
    Per-date normalization across the universe.

    Methods:
    - zscore: (x - μ) / σ per date, with optional winsorization
    - rank: percentile rank ∈ [0, 1] per date
    - minmax: (x - min) / (max - min) per date

    NaN handling: NaN values are excluded from statistics and preserved
    in the output (they don't affect other stocks' scores).
    """

    METHODS = {"zscore", "rank", "minmax"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        norm_cfg = cfg.get("normalization", cfg)
        self._method: str = norm_cfg.get("method", "zscore")
        self._clip_std: float = norm_cfg.get("clip_std", 3.0)
        self._min_obs: int = norm_cfg.get("min_obs", 10)

        if self._method not in self.METHODS:
            raise ValueError(
                f"Unknown normalization method: {self._method}. "
                f"Choose from {self.METHODS}"
            )
        logger.info(
            "Normalizer initialized: method=%s, clip_std=%.1f, min_obs=%d",
            self._method,
            self._clip_std,
            self._min_obs,
        )

    def normalize(self, features: PanelData) -> PanelData:
        """
        Normalize all feature columns cross-sectionally (per date).

        Args:
            features: PanelData with MultiIndex[date, ticker], feature columns.

        Returns:
            PanelData with same shape, values normalized per date.
        """
        validate_panel(features, "features input")

        if self._method == "zscore":
            result = self._zscore_normalize(features)
        elif self._method == "rank":
            result = self._rank_normalize(features)
        elif self._method == "minmax":
            result = self._minmax_normalize(features)
        else:
            raise ValueError(f"Unknown method: {self._method}")

        logger.info(
            "Normalized %d features across %d dates (method=%s)",
            len(result.columns),
            result.index.get_level_values(Cols.DATE).nunique(),
            self._method,
        )
        return result

    def _zscore_normalize(self, df: PanelData) -> PanelData:
        """
        Z-score normalization per date.

        Steps:
        1. For each date, compute mean and std across tickers
        2. z = (x - mean) / std
        3. Clip to ±clip_std
        4. Skip dates with fewer than min_obs valid observations
        """
        grouped = df.groupby(level=Cols.DATE)

        mean = grouped.transform("mean")
        std = grouped.transform("std")
        count = grouped.transform("count")

        # Mask dates with insufficient observations
        insufficient = count < self._min_obs
        std = std.replace(0, np.nan)

        zscore = (df - mean) / std
        zscore = zscore.clip(lower=-self._clip_std, upper=self._clip_std)
        zscore[insufficient] = np.nan

        return zscore

    def _rank_normalize(self, df: PanelData) -> PanelData:
        """
        Percentile rank normalization per date.

        Output: [0, 1] where 0 = worst, 1 = best in that day's universe.
        """
        def _pct_rank(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) < self._min_obs:
                return group * np.nan
            return group.rank(pct=True)

        return df.groupby(level=Cols.DATE).transform(_pct_rank)

    def _minmax_normalize(self, df: PanelData) -> PanelData:
        """Min-max normalization per date to [0, 1]."""
        grouped = df.groupby(level=Cols.DATE)
        min_val = grouped.transform("min")
        max_val = grouped.transform("max")
        range_val = (max_val - min_val).replace(0, np.nan)

        count = grouped.transform("count")
        result = (df - min_val) / range_val
        result[count < self._min_obs] = np.nan

        return result
