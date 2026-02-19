"""Tests for Feature computation and registry."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Cols
from features.registry import FeatureRegistry


@pytest.fixture
def sample_ticker_df():
    """Single-ticker OHLCV DataFrame indexed by date."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.DataFrame(
        {
            Cols.OPEN: close + np.random.randn(100),
            Cols.HIGH: close + abs(np.random.randn(100)) * 2,
            Cols.LOW: close - abs(np.random.randn(100)) * 2,
            Cols.CLOSE: close,
            Cols.VOLUME: np.random.randint(10000, 500000, 100),
        },
        index=dates,
    )


class TestFeatureRegistry:
    def test_registry_is_singleton(self):
        r1 = FeatureRegistry()
        r2 = FeatureRegistry()
        assert r1 is r2

    def test_features_registered_after_import(self):
        # Import triggers @feature decorators
        import features.technical  # noqa: F401
        import features.microstructure  # noqa: F401

        registry = FeatureRegistry()
        assert len(registry.all_features) > 0
        assert "momentum_5d" in registry.all_features
        assert "rsi_14" in registry.all_features
        assert "volume_ratio_5d" in registry.all_features


class TestTechnicalFeatures:
    def test_momentum_5d(self, sample_ticker_df):
        import features.technical as tech
        result = tech.momentum_5d(sample_ticker_df)
        assert len(result) == len(sample_ticker_df)
        # First 5 values should be NaN
        assert result.iloc[:5].isna().all()
        # Remaining should be numeric
        assert result.iloc[5:].notna().all()

    def test_rsi_14(self, sample_ticker_df):
        import features.technical as tech
        result = tech.rsi_14(sample_ticker_df)
        # RSI should be between 0 and 100
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_volatility_20d(self, sample_ticker_df):
        import features.technical as tech
        result = tech.volatility_20d(sample_ticker_df)
        valid = result.dropna()
        # Volatility should be non-negative
        assert (valid >= 0).all()
