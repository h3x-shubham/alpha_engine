"""
Trading calendar and date utilities aligned to NSE holidays.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("alpha_engine.dates")

# Major NSE holidays (non-exhaustive; extend or load from file for production)
_NSE_HOLIDAYS = [
    "2024-01-26", "2024-03-25", "2024-03-29", "2024-04-11",
    "2024-04-14", "2024-04-17", "2024-04-21", "2024-05-01",
    "2024-05-23", "2024-06-17", "2024-07-17", "2024-08-15",
    "2024-09-16", "2024-10-02", "2024-10-12", "2024-10-31",
    "2024-11-01", "2024-11-15", "2024-12-25",
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31",
    "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01",
    "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-20",
    "2025-10-21", "2025-11-05", "2025-11-26", "2025-12-25",
]


def get_nse_holidays() -> list[pd.Timestamp]:
    """Return list of known NSE holidays as Timestamps."""
    return [pd.Timestamp(d) for d in _NSE_HOLIDAYS]


def get_trading_dates(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    holidays: Sequence[pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    """
    Generate a DatetimeIndex of trading days between start and end.

    Excludes weekends (Saturday/Sunday) and NSE holidays.
    """
    if holidays is None:
        holidays = get_nse_holidays()
    all_days = pd.bdate_range(start=start, end=end, freq="B")
    holiday_set = set(pd.Timestamp(h).normalize() for h in holidays)
    trading_days = all_days[~all_days.isin(holiday_set)]
    logger.debug(
        "Trading dates: %d days from %s to %s", len(trading_days), start, end
    )
    return trading_days


def offset_trading_days(
    date: pd.Timestamp,
    offset: int,
    trading_dates: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """
    Offset a date by N trading days within a known calendar.

    Args:
        date: Starting date.
        offset: Number of trading days to move (positive = forward).
        trading_dates: Sorted index of valid trading dates.

    Returns:
        The offset date, or None if out of range.
    """
    date = pd.Timestamp(date).normalize()
    if date not in trading_dates:
        # Snap to nearest trading date
        idx = trading_dates.searchsorted(date)
        if idx >= len(trading_dates):
            return None
        date = trading_dates[idx]

    loc = trading_dates.get_loc(date)
    target = loc + offset
    if 0 <= target < len(trading_dates):
        return trading_dates[target]
    return None


def split_date_range(
    trading_dates: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    embargo_days: int = 0,
    expanding: bool = False,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate walk-forward train/test date splits.

    Args:
        trading_dates: Full sorted calendar.
        train_days: Number of training days per window.
        test_days: Number of test days per window.
        embargo_days: Gap between train end and test start.
        expanding: If True, training window expands from the start.

    Returns:
        List of (train_dates, test_dates) tuples.
    """
    splits = []
    n = len(trading_dates)
    cursor = train_days

    while cursor + embargo_days + test_days <= n:
        if expanding:
            train_start = 0
        else:
            train_start = cursor - train_days

        train_end = cursor
        test_start = cursor + embargo_days
        test_end = min(test_start + test_days, n)

        train_idx = trading_dates[train_start:train_end]
        test_idx = trading_dates[test_start:test_end]
        splits.append((train_idx, test_idx))

        cursor = test_end

    logger.info("Generated %d walk-forward splits", len(splits))
    return splits
