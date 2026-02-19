"""
Abstract base class for all models in the alpha engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """
    Abstract model interface.

    All models must implement fit, predict, save, and load.
    This allows swapping LightGBM for other models (XGBoost,
    neural nets, etc.) without changing the pipeline.
    """

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Feature matrix (rows = samples, cols = features).
            y_train: Target vector.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets.

        Returns:
            Dict of training metrics (e.g., best iteration, train loss).
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Series of predicted values (same index as X).
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        ...

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Return feature importances, sorted descending."""
        ...
