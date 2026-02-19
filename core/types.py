"""
Shared type aliases and enums for the alpha engine.

Convention: All intermediate DataFrames use MultiIndex[date, ticker]
(the PanelData type alias). This ensures a consistent interface across
all modules and enables efficient cross-sectional operations via
groupby(level='date').
"""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import pandas as pd


# ── Panel Data Convention ─────────────────────
# Every DataFrame flowing between modules uses this shape:
#   Index  : MultiIndex with levels ['date', 'ticker']
#   Columns: feature names, price fields, or signal scores
PanelData: TypeAlias = pd.DataFrame


class ReturnType(str, Enum):
    """Forward return computation method."""
    LOG = "log"
    SIMPLE = "simple"


class SignalDirection(str, Enum):
    """Direction of an alpha signal."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class RunMode(str, Enum):
    """Pipeline execution mode."""
    RESEARCH = "research"
    LIVE = "live"


# ── Column Name Constants ─────────────────────
class Cols:
    """Standardized column names used across the pipeline."""
    DATE = "date"
    TICKER = "ticker"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    VWAP = "vwap"
    FORWARD_RETURN = "fwd_return"
    PREDICTION = "prediction"
    SIGNAL = "signal"
    WEIGHT = "weight"


def make_panel_index(dates: pd.Series, tickers: pd.Series) -> pd.MultiIndex:
    """Create a standardized MultiIndex from date and ticker columns."""
    return pd.MultiIndex.from_arrays([dates, tickers], names=[Cols.DATE, Cols.TICKER])


def validate_panel(df: PanelData, name: str = "DataFrame") -> None:
    """Validate that a DataFrame conforms to PanelData conventions."""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"{name} must have a MultiIndex, got {type(df.index)}")
    if list(df.index.names) != [Cols.DATE, Cols.TICKER]:
        raise ValueError(
            f"{name} MultiIndex levels must be ['date', 'ticker'], "
            f"got {df.index.names}"
        )
