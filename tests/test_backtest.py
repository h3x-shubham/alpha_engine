"""Tests for BacktestEngine and metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Cols
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics


@pytest.fixture
def sample_data():
    """Create synthetic predictions and OHLCV for backtesting."""
    dates = pd.bdate_range("2024-01-01", periods=60)
    tickers = [f"S{i}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=[Cols.DATE, Cols.TICKER])
    np.random.seed(42)
    n = len(idx)

    predictions = pd.DataFrame(
        {"prediction": np.random.randn(n)}, index=idx
    )
    ohlcv = pd.DataFrame(
        {
            Cols.CLOSE: 100 + np.cumsum(np.random.randn(n) * 0.5),
            Cols.VOLUME: np.random.randint(10000, 100000, n),
        },
        index=idx,
    )
    return predictions, ohlcv


class TestBacktestEngine:
    def test_backtest_runs(self, sample_data):
        predictions, ohlcv = sample_data
        config = {"backtest": {"initial_capital": 1_000_000, "top_n": 5, "rebalance_frequency_days": 5}}
        engine = BacktestEngine(config)
        results = engine.run(predictions, ohlcv)

        assert "equity_curve" in results
        assert "daily_returns" in results
        assert len(results["equity_curve"]) > 0

    def test_equity_starts_at_initial_capital(self, sample_data):
        predictions, ohlcv = sample_data
        config = {"backtest": {"initial_capital": 1_000_000, "top_n": 5, "rebalance_frequency_days": 5}}
        engine = BacktestEngine(config)
        results = engine.run(predictions, ohlcv)

        # First non-zero equity should be close to initial capital
        equity = results["equity_curve"]
        assert abs(equity.iloc[0] - 1_000_000) < 100_000


class TestMetrics:
    def test_metrics_from_returns(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)
        metrics = compute_metrics(returns)

        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "total_return_pct" in metrics
        assert "hit_rate_pct" in metrics
        assert metrics["n_trading_days"] == 252

    def test_empty_returns(self):
        metrics = compute_metrics(pd.Series(dtype=float))
        assert metrics == {}
