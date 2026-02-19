"""
Technical indicator features.

Each function is decorated with @feature to self-register.
Functions accept a single-ticker DataFrame (sorted by date, indexed
by date) with OHLCV columns and return a Series of feature values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import Cols
from features.registry import feature


# ── Momentum ──────────────────────────────────

@feature("momentum_5d", lookback=5, category="technical")
def momentum_5d(df: pd.DataFrame) -> pd.Series:
    """5-day log return."""
    return np.log(df[Cols.CLOSE] / df[Cols.CLOSE].shift(5))


@feature("momentum_10d", lookback=10, category="technical")
def momentum_10d(df: pd.DataFrame) -> pd.Series:
    """10-day log return."""
    return np.log(df[Cols.CLOSE] / df[Cols.CLOSE].shift(10))


@feature("momentum_20d", lookback=20, category="technical")
def momentum_20d(df: pd.DataFrame) -> pd.Series:
    """20-day log return."""
    return np.log(df[Cols.CLOSE] / df[Cols.CLOSE].shift(20))


@feature("momentum_60d", lookback=60, category="technical")
def momentum_60d(df: pd.DataFrame) -> pd.Series:
    """60-day log return."""
    return np.log(df[Cols.CLOSE] / df[Cols.CLOSE].shift(60))


# ── Volatility ────────────────────────────────

@feature("volatility_20d", lookback=20, category="technical")
def volatility_20d(df: pd.DataFrame) -> pd.Series:
    """20-day rolling standard deviation of daily log returns."""
    daily_ret = np.log(df[Cols.CLOSE] / df[Cols.CLOSE].shift(1))
    return daily_ret.rolling(20).std()


@feature("volatility_60d", lookback=60, category="technical")
def volatility_60d(df: pd.DataFrame) -> pd.Series:
    """60-day rolling standard deviation of daily log returns."""
    daily_ret = np.log(df[Cols.CLOSE] / df[Cols.CLOSE].shift(1))
    return daily_ret.rolling(60).std()


# ── RSI ───────────────────────────────────────

@feature("rsi_14", lookback=14, category="technical")
def rsi_14(df: pd.DataFrame) -> pd.Series:
    """14-period Relative Strength Index."""
    delta = df[Cols.CLOSE].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ──────────────────────────────────────

@feature("macd_signal", lookback=26, category="technical")
def macd_signal(df: pd.DataFrame) -> pd.Series:
    """MACD histogram (MACD line - signal line)."""
    ema12 = df[Cols.CLOSE].ewm(span=12, adjust=False).mean()
    ema26 = df[Cols.CLOSE].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line


# ── Bollinger ─────────────────────────────────

@feature("bollinger_width", lookback=20, category="technical")
def bollinger_width(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band width: (upper - lower) / middle."""
    sma = df[Cols.CLOSE].rolling(20).mean()
    std = df[Cols.CLOSE].rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / sma.replace(0, np.nan)


# ── ATR ───────────────────────────────────────

@feature("atr_14", lookback=14, category="technical")
def atr_14(df: pd.DataFrame) -> pd.Series:
    """14-period Average True Range, normalized by close price."""
    high = df[Cols.HIGH]
    low = df[Cols.LOW]
    close_prev = df[Cols.CLOSE].shift(1)

    tr = pd.concat(
        [high - low, (high - close_prev).abs(), (low - close_prev).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(14).mean()
    # Normalize by close to make cross-sectionally comparable
    return atr / df[Cols.CLOSE].replace(0, np.nan)


# ── Mean Reversion ────────────────────────────

@feature("mean_reversion_20d", lookback=20, category="technical")
def mean_reversion_20d(df: pd.DataFrame) -> pd.Series:
    """Distance of close from 20-day SMA, in units of 20d std."""
    sma = df[Cols.CLOSE].rolling(20).mean()
    std = df[Cols.CLOSE].rolling(20).std()
    return (df[Cols.CLOSE] - sma) / std.replace(0, np.nan)
