"""
Signal-to-PnL backtest engine.

Takes out-of-sample predictions from the walk-forward engine, converts
them to portfolio weights, and simulates daily PnL.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.types import Cols, PanelData, validate_panel

logger = logging.getLogger("alpha_engine.backtest.engine")


class BacktestEngine:
    """
    Replays model predictions through a simple portfolio construction
    and PnL calculation.

    Strategy:
    - Each rebalance day: rank stocks by prediction score
    - Go long the top-N stocks with equal weight
    - Hold for rebalance_frequency_days
    - Compute daily PnL from actual returns
    """

    def __init__(self, config: dict[str, Any]) -> None:
        bt_cfg = config.get("backtest", {})
        self._initial_capital: float = bt_cfg.get("initial_capital", 1_000_000.0)
        self._top_n: int = bt_cfg.get("top_n", 10)
        self._rebalance_freq: int = bt_cfg.get("rebalance_frequency_days", 5)
        self._long_only: bool = bt_cfg.get("long_only", True)

        # Results
        self._daily_pnl: pd.Series | None = None
        self._equity_curve: pd.Series | None = None
        self._positions: pd.DataFrame | None = None

    def run(
        self,
        predictions: PanelData,
        ohlcv: PanelData,
    ) -> dict[str, Any]:
        """
        Run the backtest.

        Args:
            predictions: PanelData with 'prediction' column [date × ticker].
            ohlcv: PanelData with 'close' column [date × ticker].

        Returns:
            Dict with 'equity_curve', 'daily_returns', 'positions'.
        """
        pred_col = "prediction" if "prediction" in predictions.columns else predictions.columns[0]

        # Compute daily returns from OHLCV
        close = ohlcv[Cols.CLOSE].unstack(level=Cols.TICKER)
        daily_returns = close.pct_change().shift(-1)  # Next-day return

        # Get rebalance dates
        pred_dates = predictions.index.get_level_values(Cols.DATE).unique().sort_values()
        rebalance_dates = pred_dates[:: self._rebalance_freq]

        logger.info(
            "Backtest: %d prediction dates, %d rebalance dates, top-%d strategy",
            len(pred_dates),
            len(rebalance_dates),
            self._top_n,
        )

        # Build position weights on each rebalance date
        all_weights: dict[pd.Timestamp, dict[str, float]] = {}

        for reb_date in rebalance_dates:
            try:
                day_preds = predictions.loc[reb_date, pred_col]
            except KeyError:
                continue

            if isinstance(day_preds, (int, float)):
                continue

            # Drop NaN predictions
            day_preds = day_preds.dropna()
            if len(day_preds) < self._top_n:
                continue

            # Rank and pick top-N
            ranked = day_preds.sort_values(ascending=False)
            top_tickers = ranked.head(self._top_n).index.tolist()

            # Equal weight
            weight = 1.0 / self._top_n
            weights = {ticker: weight for ticker in top_tickers}
            all_weights[reb_date] = weights

        if not all_weights:
            logger.warning("No valid rebalance dates — backtest empty")
            return {"equity_curve": pd.Series(dtype=float), "daily_returns": pd.Series(dtype=float)}

        # Simulate daily PnL
        portfolio_returns = []
        current_weights: dict[str, float] = {}

        for date in pred_dates:
            # Update weights on rebalance dates
            if date in all_weights:
                current_weights = all_weights[date]

            if not current_weights:
                portfolio_returns.append((date, 0.0))
                continue

            # Weighted daily return
            day_ret = 0.0
            for ticker, w in current_weights.items():
                if ticker in daily_returns.columns and date in daily_returns.index:
                    r = daily_returns.loc[date, ticker]
                    if not np.isnan(r):
                        day_ret += w * r

            portfolio_returns.append((date, day_ret))

        # Build equity curve
        ret_series = pd.Series(
            {date: ret for date, ret in portfolio_returns},
            name="portfolio_return",
        )
        equity = (1 + ret_series).cumprod() * self._initial_capital
        equity.name = "equity"

        self._daily_pnl = ret_series
        self._equity_curve = equity

        logger.info(
            "Backtest complete: final equity=%.0f, total return=%.2f%%",
            equity.iloc[-1] if len(equity) > 0 else 0,
            (equity.iloc[-1] / self._initial_capital - 1) * 100 if len(equity) > 0 else 0,
        )

        return {
            "equity_curve": equity,
            "daily_returns": ret_series,
            "n_rebalances": len(all_weights),
        }

    @property
    def equity_curve(self) -> pd.Series | None:
        return self._equity_curve

    @property
    def daily_returns(self) -> pd.Series | None:
        return self._daily_pnl
