"""Tests for walk-forward date splits."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.date_utils import split_date_range, get_trading_dates


class TestWalkForwardSplits:
    @pytest.fixture
    def trading_dates(self):
        return get_trading_dates("2023-01-01", "2024-12-31")

    def test_splits_non_empty(self, trading_dates):
        splits = split_date_range(
            trading_dates, train_days=252, test_days=21, embargo_days=5
        )
        assert len(splits) > 0

    def test_no_overlap_train_test(self, trading_dates):
        splits = split_date_range(
            trading_dates, train_days=252, test_days=21, embargo_days=5
        )
        for train_idx, test_idx in splits:
            # No date should appear in both train and test
            overlap = train_idx.intersection(test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_embargo_gap(self, trading_dates):
        embargo = 5
        splits = split_date_range(
            trading_dates, train_days=252, test_days=21, embargo_days=embargo
        )
        for train_idx, test_idx in splits:
            train_end = train_idx[-1]
            test_start = test_idx[0]
            # Test start should be at least embargo days after train end
            gap = trading_dates.get_loc(test_start) - trading_dates.get_loc(train_end)
            assert gap >= embargo, f"Gap {gap} < embargo {embargo}"

    def test_temporal_ordering(self, trading_dates):
        splits = split_date_range(
            trading_dates, train_days=252, test_days=21, embargo_days=5
        )
        for train_idx, test_idx in splits:
            assert train_idx[-1] < test_idx[0], "Train must end before test starts"

    def test_expanding_window(self, trading_dates):
        splits = split_date_range(
            trading_dates, train_days=252, test_days=21, expanding=True
        )
        if len(splits) >= 2:
            # Second fold's training set should be larger than first
            assert len(splits[1][0]) >= len(splits[0][0])
