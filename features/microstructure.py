"""
Microstructure features: volume, liquidity, and spread indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import Cols
from features.registry import feature


@feature("volume_ratio_5d", lookback=5, category="microstructure")
def volume_ratio_5d(df: pd.DataFrame) -> pd.Series:
    """Today's volume / 5-day average volume."""
    avg_vol = df[Cols.VOLUME].rolling(5).mean()
    return df[Cols.VOLUME] / avg_vol.replace(0, np.nan)


@feature("volume_ratio_20d", lookback=20, category="microstructure")
def volume_ratio_20d(df: pd.DataFrame) -> pd.Series:
    """Today's volume / 20-day average volume."""
    avg_vol = df[Cols.VOLUME].rolling(20).mean()
    return df[Cols.VOLUME] / avg_vol.replace(0, np.nan)


@feature("amihud_illiquidity_20d", lookback=20, category="microstructure")
def amihud_illiquidity_20d(df: pd.DataFrame) -> pd.Series:
    """
    20-day Amihud illiquidity ratio.

    Amihud = mean(|daily_return| / volume) over lookback period.
    Higher values indicate less liquid stocks.
    """
    daily_ret = np.log(df[Cols.CLOSE] / df[Cols.CLOSE].shift(1)).abs()
    volume = df[Cols.VOLUME].replace(0, np.nan)
    illiq = daily_ret / volume
    return illiq.rolling(20).mean()


@feature("vwap_deviation", lookback=1, category="microstructure")
def vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """
    Close price deviation from estimated VWAP.

    If VWAP column is not available, estimates using typical price:
    VWAP_proxy = (high + low + close) / 3
    """
    if Cols.VWAP in df.columns:
        vwap = df[Cols.VWAP]
    else:
        vwap = (df[Cols.HIGH] + df[Cols.LOW] + df[Cols.CLOSE]) / 3

    return (df[Cols.CLOSE] - vwap) / vwap.replace(0, np.nan)


@feature("high_low_spread", lookback=1, category="microstructure")
def high_low_spread(df: pd.DataFrame) -> pd.Series:
    """
    Daily high-low spread as a fraction of close.

    Proxy for intraday volatility / bid-ask spread.
    """
    return (df[Cols.HIGH] - df[Cols.LOW]) / df[Cols.CLOSE].replace(0, np.nan)
