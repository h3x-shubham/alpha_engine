"""
Walk-forward validation engine.

Trains the model on expanding or rolling windows of historical data,
with an embargo gap to prevent label leakage, and collects out-of-sample
predictions for fair backtesting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.types import Cols, PanelData
from core.date_utils import split_date_range
from models.base import BaseModel

logger = logging.getLogger("alpha_engine.models.walk_forward")


class WalkForwardEngine:
    """
    Walk-forward (rolling/expanding window) validation.

    For each fold:
    1. Train on [train_start, train_end)
    2. Skip embargo gap
    3. Predict on [test_start, test_end)
    4. Collect out-of-sample predictions

    The collected OOS predictions cover the full test period and can
    be directly fed into the backtest engine.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        wf_cfg = config.get("walk_forward", {})
        self._train_days: int = wf_cfg.get("train_window_days", 504)
        self._test_days: int = wf_cfg.get("test_window_days", 21)
        self._embargo_days: int = wf_cfg.get("embargo_days", 5)
        self._expanding: bool = wf_cfg.get("expanding", False)

        # Collected results
        self._oos_predictions: list[pd.Series] = []
        self._fold_metrics: list[dict[str, Any]] = []

        logger.info(
            "WalkForward: train=%d, test=%d, embargo=%d, expanding=%s",
            self._train_days,
            self._test_days,
            self._embargo_days,
            self._expanding,
        )

    def run(
        self,
        model: BaseModel,
        features: PanelData,
        labels: pd.Series,
    ) -> PanelData:
        """
        Execute walk-forward validation.

        Args:
            model: Model implementing BaseModel interface.
            features: PanelData [date × ticker] with feature columns.
            labels: Series [date × ticker] with forward return labels.

        Returns:
            PanelData with 'prediction' column covering all OOS periods.
        """
        # Get trading dates from the feature panel
        all_dates = features.index.get_level_values(Cols.DATE).unique().sort_values()
        trading_dates = pd.DatetimeIndex(all_dates)

        # Generate splits
        splits = split_date_range(
            trading_dates=trading_dates,
            train_days=self._train_days,
            test_days=self._test_days,
            embargo_days=self._embargo_days,
            expanding=self._expanding,
        )

        if not splits:
            raise ValueError(
                f"No valid walk-forward splits. Need at least "
                f"{self._train_days + self._embargo_days + self._test_days} dates, "
                f"have {len(trading_dates)}."
            )

        logger.info("Running %d walk-forward folds", len(splits))
        self._oos_predictions = []
        self._fold_metrics = []

        for fold_idx, (train_dates, test_dates) in enumerate(splits):
            logger.info(
                "Fold %d/%d: train [%s → %s] (%d days), test [%s → %s] (%d days)",
                fold_idx + 1,
                len(splits),
                train_dates[0].strftime("%Y-%m-%d"),
                train_dates[-1].strftime("%Y-%m-%d"),
                len(train_dates),
                test_dates[0].strftime("%Y-%m-%d"),
                test_dates[-1].strftime("%Y-%m-%d"),
                len(test_dates),
            )

            # Extract train data
            train_mask = features.index.get_level_values(Cols.DATE).isin(train_dates)
            X_train = features[train_mask]
            y_train = labels[train_mask]

            # Extract test data
            test_mask = features.index.get_level_values(Cols.DATE).isin(test_dates)
            X_test = features[test_mask]

            # Train
            train_metrics = model.fit(X_train, y_train)

            # Predict OOS
            preds = model.predict(X_test)
            preds.name = "prediction"
            self._oos_predictions.append(preds)

            # Record fold metrics
            fold_info = {
                "fold": fold_idx,
                "train_start": str(train_dates[0].date()),
                "train_end": str(train_dates[-1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                **train_metrics,
            }
            self._fold_metrics.append(fold_info)

        # Combine OOS predictions
        all_preds = pd.concat(self._oos_predictions)

        # Remove any duplicates (overlap between folds)
        all_preds = all_preds[~all_preds.index.duplicated(keep="first")]

        logger.info(
            "Walk-forward complete: %d total OOS predictions across %d folds",
            len(all_preds),
            len(splits),
        )
        return all_preds.to_frame()

    @property
    def fold_metrics(self) -> list[dict[str, Any]]:
        """Return per-fold training metrics."""
        return self._fold_metrics

    @property
    def n_folds(self) -> int:
        """Number of completed folds."""
        return len(self._fold_metrics)
