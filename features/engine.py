"""
Cross-sectional feature computation engine.

Iterates date-by-date over the universe panel. For each date, computes
all registered features across all tickers simultaneously, producing a
[date × ticker × feature] result.

This is the core abstraction for cross-sectional daily processing.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.types import Cols, PanelData, validate_panel
from features.registry import FeatureRegistry, FeatureMeta

logger = logging.getLogger("alpha_engine.features.engine")


class FeatureEngine:
    """
    Cross-sectional feature computation engine.

    For each trading date:
    1. Select the universe of tickers available on that date
    2. Compute all enabled features using each ticker's history
    3. Output a panel [date × ticker] with feature columns

    This structure mirrors live production: on any given day you have
    a set of tickers and compute signals across all of them.
    """

    def __init__(self, feature_config: dict[str, Any]) -> None:
        self._registry = FeatureRegistry()
        self._feature_config = feature_config
        self._enabled_features: list[FeatureMeta] = []

    def initialize(self) -> None:
        """Resolve which features are enabled from config."""
        # Import feature modules to trigger @feature registration
        import features.technical  # noqa: F401
        import features.microstructure  # noqa: F401

        self._enabled_features = self._registry.get_enabled(self._feature_config)
        logger.info(
            "Feature engine initialized: %d features enabled",
            len(self._enabled_features),
        )

    def compute(self, ohlcv: PanelData) -> PanelData:
        """
        Compute all enabled features on the OHLCV panel.

        Args:
            ohlcv: PanelData with MultiIndex[date, ticker] and OHLCV columns.

        Returns:
            PanelData with feature columns appended. The first
            max_lookback dates will have NaN features and are dropped.
        """
        validate_panel(ohlcv, "OHLCV input")

        if not self._enabled_features:
            self.initialize()

        features_list: list[pd.DataFrame] = []
        tickers = ohlcv.index.get_level_values(Cols.TICKER).unique()

        logger.info(
            "Computing %d features for %d tickers",
            len(self._enabled_features),
            len(tickers),
        )

        for ticker in tickers:
            # Extract single-ticker time series (sorted by date)
            ticker_data = ohlcv.xs(ticker, level=Cols.TICKER).sort_index()

            ticker_features = pd.DataFrame(index=ticker_data.index)

            for feat_meta in self._enabled_features:
                try:
                    values = feat_meta.compute_fn(ticker_data)
                    ticker_features[feat_meta.name] = values
                except Exception as e:
                    logger.warning(
                        "Feature %s failed for %s: %s",
                        feat_meta.name,
                        ticker,
                        e,
                    )
                    ticker_features[feat_meta.name] = float("nan")

            # Re-attach ticker to index
            ticker_features[Cols.TICKER] = ticker
            ticker_features.index.name = Cols.DATE
            ticker_features = ticker_features.reset_index()
            features_list.append(ticker_features)

        if not features_list:
            return pd.DataFrame()

        result = pd.concat(features_list, ignore_index=True)
        result = result.set_index([Cols.DATE, Cols.TICKER]).sort_index()

        # Drop warmup period
        max_lookback = self._registry.max_lookback
        dates = result.index.get_level_values(Cols.DATE).unique().sort_values()
        if len(dates) > max_lookback:
            valid_dates = dates[max_lookback:]
            result = result.loc[result.index.get_level_values(Cols.DATE).isin(valid_dates)]

        n_nan = result.isna().sum().sum()
        logger.info(
            "Features computed: %d rows × %d features (%d NaN cells)",
            len(result),
            len(result.columns),
            n_nan,
        )
        return result

    @property
    def feature_names(self) -> list[str]:
        """Return sorted list of enabled feature names."""
        return sorted(f.name for f in self._enabled_features)
