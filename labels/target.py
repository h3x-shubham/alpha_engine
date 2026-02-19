"""
Forward return label construction.

Builds N-day forward return labels aligned to the panel index,
taking care to avoid look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.types import Cols, PanelData, validate_panel

logger = logging.getLogger("alpha_engine.labels.target")


class TargetBuilder:
    """
    Constructs forward return labels for cross-sectional prediction.

    For each (date, ticker) pair, computes the return from date to
    date + horizon_days. The label at date t uses price information
    from t+1 to t+horizon — this is what we're predicting.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        label_cfg = config.get("labels", {})
        self._horizon: int = label_cfg.get("horizon_days", 5)
        self._return_type: str = label_cfg.get("return_type", "log")
        logger.info(
            "TargetBuilder: horizon=%d days, return_type=%s",
            self._horizon,
            self._return_type,
        )

    def build(self, ohlcv: PanelData) -> pd.Series:
        """
        Compute forward returns for all (date, ticker) pairs.

        Args:
            ohlcv: PanelData with at least a 'close' column.

        Returns:
            Series indexed by [date, ticker] with forward return values.
            The last `horizon` dates will have NaN (no future data).
        """
        validate_panel(ohlcv, "OHLCV for labels")

        if Cols.CLOSE not in ohlcv.columns:
            raise ValueError(f"OHLCV must contain '{Cols.CLOSE}' column")

        close = ohlcv[Cols.CLOSE]

        # Compute forward return per ticker
        def _forward_return(group: pd.Series) -> pd.Series:
            future_close = group.shift(-self._horizon)
            if self._return_type == "log":
                return np.log(future_close / group)
            else:
                return (future_close - group) / group

        labels = close.groupby(level=Cols.TICKER).transform(_forward_return)
        labels.name = Cols.FORWARD_RETURN

        n_valid = labels.notna().sum()
        n_total = len(labels)
        logger.info(
            "Labels built: %d valid / %d total (%.1f%% coverage)",
            n_valid,
            n_total,
            100 * n_valid / max(n_total, 1),
        )
        return labels

    def build_panel(self, ohlcv: PanelData) -> PanelData:
        """Build labels and return as a single-column PanelData."""
        labels = self.build(ohlcv)
        return labels.to_frame()
