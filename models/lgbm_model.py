"""
LightGBM model wrapper.

Implements the BaseModel interface for LightGBM regressor/ranker,
with support for early stopping, custom loss, and feature importance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.base import BaseModel

logger = logging.getLogger("alpha_engine.models.lgbm")


class LightGBMModel(BaseModel):
    """
    LightGBM wrapper for cross-sectional return prediction.

    Supports:
    - Regression objective (MSE / MAE)
    - LambdaRank objective (for ranking stocks by expected return)
    - Early stopping on validation set
    - SHAP-compatible feature importance
    """

    def __init__(self, config: dict[str, Any]) -> None:
        model_cfg = config.get("model", config)
        self._objective: str = model_cfg.get("objective", "regression")
        self._params: dict[str, Any] = model_cfg.get("hyperparameters", {})
        self._model = None

        # Build LightGBM parameter dict
        self._lgb_params = {
            "objective": self._objective,
            "n_estimators": self._params.get("n_estimators", 500),
            "max_depth": self._params.get("max_depth", 6),
            "learning_rate": self._params.get("learning_rate", 0.05),
            "num_leaves": self._params.get("num_leaves", 31),
            "min_child_samples": self._params.get("min_child_samples", 20),
            "subsample": self._params.get("subsample", 0.8),
            "colsample_bytree": self._params.get("colsample_bytree", 0.8),
            "reg_alpha": self._params.get("reg_alpha", 0.1),
            "reg_lambda": self._params.get("reg_lambda", 1.0),
            "random_state": self._params.get("random_state", 42),
            "verbose": -1,
            "n_jobs": -1,
        }

        logger.info(
            "LightGBM model initialized: objective=%s, n_estimators=%d, lr=%.4f",
            self._objective,
            self._lgb_params["n_estimators"],
            self._lgb_params["learning_rate"],
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Train the LightGBM model with optional early stopping."""
        import lightgbm as lgb

        self._model = lgb.LGBMRegressor(**self._lgb_params)

        fit_kwargs: dict[str, Any] = {}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        # Drop NaN rows from training data
        train_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_clean = X_train[train_mask]
        y_clean = y_train[train_mask]

        logger.info(
            "Training LightGBM: %d samples, %d features",
            len(X_clean),
            X_clean.shape[1],
        )

        self._model.fit(X_clean, y_clean, **fit_kwargs)

        metrics = {
            "n_samples": len(X_clean),
            "n_features": X_clean.shape[1],
            "best_iteration": getattr(self._model, "best_iteration_", -1),
        }

        logger.info("Training complete: %s", metrics)
        return metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for the feature matrix."""
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        # Handle NaN in prediction inputs
        valid_mask = X.notna().all(axis=1)
        predictions = pd.Series(np.nan, index=X.index, name="prediction")

        if valid_mask.any():
            preds = self._model.predict(X[valid_mask])
            predictions[valid_mask] = preds

        return predictions

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(path))
        logger.info("Model saved: %s", path)

    def load(self, path: str | Path) -> None:
        """Load a model from disk."""
        import lightgbm as lgb

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        booster = lgb.Booster(model_file=str(path))
        self._model = lgb.LGBMRegressor(**self._lgb_params)
        self._model._Booster = booster
        self._model.fitted_ = True
        logger.info("Model loaded: %s", path)

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        if self._model is None:
            raise RuntimeError("Model has not been trained.")
        importances = pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_name_,
            name="importance",
        )
        return importances.sort_values(ascending=False)
