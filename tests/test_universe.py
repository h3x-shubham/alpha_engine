"""Tests for Universe class."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.universe import Universe
from core.types import Cols


@pytest.fixture
def sample_config():
    return {
        "universe": {
            "tickers": ["RELIANCE", "TCS", "INFY", "SBIN", "HDFC"],
            "exchange": "NSE",
            "min_avg_volume": 100,
            "min_price": 10.0,
            "max_tickers": 50,
        }
    }


@pytest.fixture
def sample_ohlcv():
    """Create synthetic OHLCV panel data."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    tickers = ["RELIANCE", "TCS", "INFY", "SBIN", "HDFC"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=[Cols.DATE, Cols.TICKER])
    np.random.seed(42)
    n = len(idx)
    data = {
        Cols.OPEN: np.random.uniform(100, 500, n),
        Cols.HIGH: np.random.uniform(100, 550, n),
        Cols.LOW: np.random.uniform(80, 500, n),
        Cols.CLOSE: np.random.uniform(100, 500, n),
        Cols.VOLUME: np.random.randint(1000, 100000, n),
    }
    return pd.DataFrame(data, index=idx)


class TestUniverse:
    def test_init(self, sample_config):
        universe = Universe(sample_config)
        assert len(universe.base_tickers) == 5
        assert "RELIANCE" in universe.base_tickers

    def test_get_tickers_before_filter(self, sample_config):
        universe = Universe(sample_config)
        tickers = universe.get_tickers(pd.Timestamp("2024-01-01"))
        assert tickers == sample_config["universe"]["tickers"]

    def test_apply_filters(self, sample_config, sample_ohlcv):
        universe = Universe(sample_config)
        universe.apply_filters(sample_ohlcv)
        dates = universe.get_all_dates()
        assert len(dates) > 0
        for date in dates:
            tickers = universe.get_tickers(date)
            assert isinstance(tickers, list)
            assert all(t in sample_config["universe"]["tickers"] for t in tickers)

    def test_filter_panel(self, sample_config, sample_ohlcv):
        universe = Universe(sample_config)
        universe.apply_filters(sample_ohlcv)
        filtered = universe.filter_panel(sample_ohlcv)
        assert len(filtered) <= len(sample_ohlcv)
