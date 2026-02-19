"""Tests for CrossSectionalNormalizer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Cols
from features.normalizer import CrossSectionalNormalizer


@pytest.fixture
def sample_features():
    """Panel features: 20 dates × 10 tickers × 3 features."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    tickers = [f"STOCK_{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=[Cols.DATE, Cols.TICKER])
    np.random.seed(42)
    n = len(idx)
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(n) * 10 + 50,
            "feature_b": np.random.randn(n) * 5 + 100,
            "feature_c": np.random.exponential(2, n),
        },
        index=idx,
    )


class TestCrossSectionalNormalizer:
    def test_zscore_mean_near_zero(self, sample_features):
        normalizer = CrossSectionalNormalizer({"normalization": {"method": "zscore", "min_obs": 5}})
        normed = normalizer.normalize(sample_features)

        # Per-date mean should be ~0
        date_means = normed.groupby(level=Cols.DATE).mean()
        assert date_means.abs().max().max() < 0.1

    def test_zscore_std_near_one(self, sample_features):
        normalizer = CrossSectionalNormalizer({"normalization": {"method": "zscore", "min_obs": 5, "clip_std": 10}})
        normed = normalizer.normalize(sample_features)

        # Per-date std should be ~1 (not exact due to clipping)
        date_stds = normed.groupby(level=Cols.DATE).std()
        assert (date_stds.mean() - 1.0).abs().max() < 0.3

    def test_rank_output_range(self, sample_features):
        normalizer = CrossSectionalNormalizer({"normalization": {"method": "rank", "min_obs": 5}})
        normed = normalizer.normalize(sample_features)

        # Rank output should be in [0, 1]
        assert normed.min().min() >= 0.0
        assert normed.max().max() <= 1.0

    def test_preserves_nan(self, sample_features):
        features = sample_features.copy()
        features.iloc[0, 0] = np.nan

        normalizer = CrossSectionalNormalizer({"normalization": {"method": "zscore", "min_obs": 5}})
        normed = normalizer.normalize(features)

        assert normed.iloc[0, 0] != normed.iloc[0, 0]  # NaN check

    def test_minmax(self, sample_features):
        normalizer = CrossSectionalNormalizer({"normalization": {"method": "minmax", "min_obs": 5}})
        normed = normalizer.normalize(sample_features)

        assert normed.min().min() >= 0.0
        assert normed.max().max() <= 1.0
