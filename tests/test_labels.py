"""Tests for TargetBuilder (forward return labels)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Cols
from labels.target import TargetBuilder


@pytest.fixture
def sample_ohlcv():
    dates = pd.bdate_range("2024-01-01", periods=30)
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=[Cols.DATE, Cols.TICKER])
    np.random.seed(42)
    n = len(idx)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({Cols.CLOSE: close}, index=idx)


class TestTargetBuilder:
    def test_label_shape(self, sample_ohlcv):
        builder = TargetBuilder({"labels": {"horizon_days": 5, "return_type": "log"}})
        labels = builder.build(sample_ohlcv)
        assert len(labels) == len(sample_ohlcv)

    def test_last_n_days_are_nan(self, sample_ohlcv):
        horizon = 5
        builder = TargetBuilder({"labels": {"horizon_days": horizon, "return_type": "log"}})
        labels = builder.build(sample_ohlcv)

        # Last `horizon` dates per ticker should be NaN
        for ticker in ["A", "B", "C"]:
            ticker_labels = labels.xs(ticker, level=Cols.TICKER)
            assert ticker_labels.iloc[-horizon:].isna().all()

    def test_no_lookahead_bias(self, sample_ohlcv):
        """Labels at date t should only use prices from t+1..t+horizon."""
        horizon = 5
        builder = TargetBuilder({"labels": {"horizon_days": horizon, "return_type": "simple"}})
        labels = builder.build(sample_ohlcv)

        # Manual check for ticker A
        close_a = sample_ohlcv.xs("A", level=Cols.TICKER)[Cols.CLOSE]
        for i in range(len(close_a) - horizon):
            expected = (close_a.iloc[i + horizon] - close_a.iloc[i]) / close_a.iloc[i]
            actual = labels.xs("A", level=Cols.TICKER).iloc[i]
            assert abs(actual - expected) < 1e-10, f"Mismatch at index {i}"
