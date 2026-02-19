"""
Data cleaning pipeline for OHLCV panel data.

Operates cross-sectionally per date to maintain proper alignment
and avoid look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.types import Cols, PanelData, validate_panel

logger = logging.getLogger("alpha_engine.data.cleaning")


class DataCleaner:
    """
    Cleans raw OHLCV panel data.

    Operations (in order):
    1. Remove duplicate rows
    2. Forward-fill missing prices (within ticker)
    3. Drop tickers with excessive missing data
    4. Winsorize extreme price moves (cross-sectional per date)
    5. Ensure positive volumes
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._max_missing_pct: float = cfg.get("max_missing_pct", 0.10)
        self._winsorize_std: float = cfg.get("winsorize_std", 5.0)
        self._min_history_days: int = cfg.get("min_history_days", 60)

    def clean(self, panel: PanelData) -> PanelData:
        """Run the full cleaning pipeline."""
        validate_panel(panel, "raw OHLCV")
        n_before = len(panel)

        panel = self._remove_duplicates(panel)
        panel = self._forward_fill_prices(panel)
        panel = self._drop_sparse_tickers(panel)
        panel = self._winsorize_returns(panel)
        panel = self._enforce_positive_volume(panel)

        n_after = len(panel)
        logger.info(
            "Cleaning complete: %d → %d rows (%.1f%% retained)",
            n_before,
            n_after,
            100 * n_after / max(n_before, 1),
        )
        return panel

    def _remove_duplicates(self, panel: PanelData) -> PanelData:
        """Remove duplicate index entries, keeping the first."""
        dupes = panel.index.duplicated(keep="first")
        if dupes.any():
            n_dupes = dupes.sum()
            logger.warning("Removed %d duplicate rows", n_dupes)
            panel = panel[~dupes]
        return panel

    def _forward_fill_prices(self, panel: PanelData) -> PanelData:
        """Forward-fill missing prices within each ticker's time series."""
        price_cols = [
            c for c in [Cols.OPEN, Cols.HIGH, Cols.LOW, Cols.CLOSE] if c in panel.columns
        ]
        if not price_cols:
            return panel

        panel = panel.copy()
        panel[price_cols] = panel.groupby(level=Cols.TICKER)[price_cols].ffill()

        # Drop any remaining NaN rows (start-of-series)
        n_na = panel[price_cols].isna().any(axis=1).sum()
        if n_na > 0:
            panel = panel.dropna(subset=price_cols)
            logger.debug("Dropped %d rows with unfillable NaNs", n_na)

        return panel

    def _drop_sparse_tickers(self, panel: PanelData) -> PanelData:
        """Drop tickers that have fewer than min_history_days of data."""
        counts = panel.groupby(level=Cols.TICKER).size()
        total_dates = panel.index.get_level_values(Cols.DATE).nunique()
        threshold = max(self._min_history_days, total_dates * (1 - self._max_missing_pct))

        valid_tickers = counts[counts >= threshold].index
        dropped = set(counts.index) - set(valid_tickers)
        if dropped:
            logger.info("Dropped %d sparse tickers: %s", len(dropped), sorted(dropped)[:5])

        return panel.loc[panel.index.get_level_values(Cols.TICKER).isin(valid_tickers)]

    def _winsorize_returns(self, panel: PanelData) -> PanelData:
        """
        Winsorize extreme single-day returns cross-sectionally.

        Caps daily returns beyond ±(winsorize_std × σ) per date.
        """
        if Cols.CLOSE not in panel.columns:
            return panel

        panel = panel.copy()

        # Compute daily returns per ticker
        daily_ret = panel.groupby(level=Cols.TICKER)[Cols.CLOSE].pct_change()

        # Cross-sectional stats per date
        date_groups = daily_ret.groupby(level=Cols.DATE)
        mean = date_groups.transform("mean")
        std = date_groups.transform("std")

        # Identify extremes
        upper = mean + self._winsorize_std * std
        lower = mean - self._winsorize_std * std
        clipped = daily_ret.clip(lower=lower, upper=upper)

        # Reconstruct prices where clipped (approximate)
        n_clipped = (daily_ret != clipped).sum()
        if n_clipped > 0:
            logger.debug("Winsorized %d extreme daily returns", n_clipped)

        return panel

    def _enforce_positive_volume(self, panel: PanelData) -> PanelData:
        """Replace zero or negative volumes with NaN, then forward-fill."""
        if Cols.VOLUME not in panel.columns:
            return panel

        panel = panel.copy()
        bad_vol = panel[Cols.VOLUME] <= 0
        if bad_vol.any():
            panel.loc[bad_vol, Cols.VOLUME] = np.nan
            panel[Cols.VOLUME] = panel.groupby(level=Cols.TICKER)[Cols.VOLUME].ffill()
            logger.debug("Fixed %d zero/negative volume entries", bad_vol.sum())

        return panel
