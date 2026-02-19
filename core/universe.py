"""
Multi-asset universe management.

Provides time-varying constituent snapshots so all downstream modules
receive (date, tickers) pairs — never hardcoded ticker lists.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.types import Cols, PanelData

logger = logging.getLogger("alpha_engine.universe")


class Universe:
    """
    Manages the investable universe of tickers with daily filtering.

    The universe can be:
    - Static: a fixed list of tickers from config
    - Dynamic: filtered daily by liquidity, price, and availability

    All downstream modules call `get_tickers(date)` to get the valid
    set of tickers for any given date, enabling proper cross-sectional
    processing over a time-varying universe.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        ucfg = config.get("universe", {})
        self._base_tickers: list[str] = ucfg.get("tickers", [])
        self._exchange: str = ucfg.get("exchange", "NSE")
        self._min_avg_volume: float = ucfg.get("min_avg_volume", 0)
        self._min_price: float = ucfg.get("min_price", 0)
        self._max_tickers: int = ucfg.get("max_tickers", 999)
        self._volume_lookback: int = 20

        # Cache: date → list of valid tickers (populated after apply_filters)
        self._daily_constituents: dict[pd.Timestamp, list[str]] = {}
        logger.info(
            "Universe initialized: %d base tickers on %s",
            len(self._base_tickers),
            self._exchange,
        )

    @property
    def base_tickers(self) -> list[str]:
        """Return the static base ticker list from config."""
        return self._base_tickers

    def apply_filters(self, ohlcv: PanelData) -> None:
        """
        Apply liquidity and price filters to build daily constituent lists.

        Args:
            ohlcv: Panel data with MultiIndex[date, ticker] and columns
                   including 'close' and 'volume'.
        """
        dates = ohlcv.index.get_level_values(Cols.DATE).unique().sort_values()
        logger.info("Applying universe filters across %d dates", len(dates))

        for date in dates:
            try:
                day_data = ohlcv.loc[date]
            except KeyError:
                continue

            mask = pd.Series(True, index=day_data.index)

            # Price filter
            if self._min_price > 0 and Cols.CLOSE in day_data.columns:
                mask &= day_data[Cols.CLOSE] >= self._min_price

            # Volume filter: use rolling average volume up to this date
            if self._min_avg_volume > 0 and Cols.VOLUME in day_data.columns:
                # For simplicity, filter on today's volume
                # (full rolling avg is computed in cleaning/feature stage)
                mask &= day_data[Cols.VOLUME] >= self._min_avg_volume

            valid_tickers = list(day_data.index[mask])

            # Restrict to base tickers if defined
            if self._base_tickers:
                valid_tickers = [t for t in valid_tickers if t in self._base_tickers]

            # Cap universe size
            valid_tickers = valid_tickers[: self._max_tickers]
            self._daily_constituents[pd.Timestamp(date)] = valid_tickers

        logger.info(
            "Universe filters applied. Avg daily constituents: %.1f",
            np.mean([len(v) for v in self._daily_constituents.values()])
            if self._daily_constituents
            else 0,
        )

    def get_tickers(self, date: pd.Timestamp) -> list[str]:
        """
        Return the valid tickers for a given date.

        If daily filters haven't been applied, returns the full base list.
        """
        if self._daily_constituents:
            return self._daily_constituents.get(pd.Timestamp(date), [])
        return self._base_tickers

    def get_all_dates(self) -> list[pd.Timestamp]:
        """Return all dates for which the universe has been computed."""
        return sorted(self._daily_constituents.keys())

    def filter_panel(self, panel: PanelData) -> PanelData:
        """
        Filter a panel DataFrame to include only valid universe members.

        Iterates date-by-date and retains only rows where the ticker
        is in that date's constituent list.
        """
        if not self._daily_constituents:
            return panel

        masks = []
        for date, tickers in self._daily_constituents.items():
            date_mask = panel.index.get_level_values(Cols.DATE) == date
            ticker_mask = panel.index.get_level_values(Cols.TICKER).isin(tickers)
            masks.append(date_mask & ticker_mask)

        if not masks:
            return panel

        combined = masks[0]
        for m in masks[1:]:
            combined |= m

        return panel.loc[combined]

    def __repr__(self) -> str:
        return (
            f"Universe(base={len(self._base_tickers)} tickers, "
            f"exchange={self._exchange}, "
            f"daily_snapshots={len(self._daily_constituents)})"
        )
