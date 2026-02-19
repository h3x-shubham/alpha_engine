"""
Performance analytics for backtest results.

Computes standard quant metrics: Sharpe, Sortino, max drawdown,
Calmar ratio, information coefficient (IC), turnover, and hit rate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from core.types import Cols

logger = logging.getLogger("alpha_engine.backtest.metrics")

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    daily_returns: pd.Series,
    predictions: pd.Series | None = None,
    actuals: pd.Series | None = None,
) -> dict[str, float]:
    """
    Compute a comprehensive set of backtest metrics.

    Args:
        daily_returns: Daily portfolio returns.
        predictions: Optional OOS predictions for IC calculation.
        actuals: Optional actual forward returns for IC calculation.

    Returns:
        Dict of metric_name → value.
    """
    metrics: dict[str, float] = {}

    if daily_returns.empty or daily_returns.isna().all():
        logger.warning("Empty or all-NaN returns — cannot compute metrics")
        return metrics

    clean_ret = daily_returns.dropna()

    # ── Return metrics ────────────────────────
    total_return = (1 + clean_ret).prod() - 1
    ann_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / max(len(clean_ret), 1)) - 1
    ann_vol = clean_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    metrics["total_return_pct"] = round(total_return * 100, 2)
    metrics["annualized_return_pct"] = round(ann_return * 100, 2)
    metrics["annualized_volatility_pct"] = round(ann_vol * 100, 2)

    # ── Sharpe ratio ──────────────────────────
    if ann_vol > 0:
        metrics["sharpe_ratio"] = round(ann_return / ann_vol, 3)
    else:
        metrics["sharpe_ratio"] = 0.0

    # ── Sortino ratio ─────────────────────────
    downside = clean_ret[clean_ret < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 0 else 0
    if downside_vol > 0:
        metrics["sortino_ratio"] = round(ann_return / downside_vol, 3)
    else:
        metrics["sortino_ratio"] = 0.0

    # ── Max drawdown ──────────────────────────
    equity = (1 + clean_ret).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    metrics["max_drawdown_pct"] = round(drawdown.min() * 100, 2)

    # ── Calmar ratio ──────────────────────────
    if metrics["max_drawdown_pct"] != 0:
        metrics["calmar_ratio"] = round(
            ann_return / abs(metrics["max_drawdown_pct"] / 100), 3
        )
    else:
        metrics["calmar_ratio"] = 0.0

    # ── Hit rate ──────────────────────────────
    positive_days = (clean_ret > 0).sum()
    metrics["hit_rate_pct"] = round(100 * positive_days / max(len(clean_ret), 1), 1)

    # ── Trading days ──────────────────────────
    metrics["n_trading_days"] = len(clean_ret)

    # ── Information Coefficient (IC) ──────────
    if predictions is not None and actuals is not None:
        ic_metrics = compute_ic(predictions, actuals)
        metrics.update(ic_metrics)

    return metrics


def compute_ic(
    predictions: pd.Series,
    actuals: pd.Series,
) -> dict[str, float]:
    """
    Compute Information Coefficient (Spearman rank correlation)
    between predictions and actual returns, per date.

    Returns:
        Dict with mean_ic, ic_std, icir (IC information ratio),
        and mean_rank_ic.
    """
    # Align indices
    common_idx = predictions.index.intersection(actuals.index)
    if len(common_idx) == 0:
        return {}

    preds = predictions.loc[common_idx]
    acts = actuals.loc[common_idx]

    # Per-date IC
    combined = pd.DataFrame({"pred": preds, "actual": acts})
    combined = combined.dropna()

    if combined.empty:
        return {}

    dates = combined.index.get_level_values(Cols.DATE).unique()
    daily_ic = []

    for date in dates:
        try:
            day_data = combined.loc[date]
            if len(day_data) < 5:
                continue
            ic, _ = stats.spearmanr(day_data["pred"], day_data["actual"])
            daily_ic.append(ic)
        except Exception:
            continue

    if not daily_ic:
        return {}

    ic_series = pd.Series(daily_ic)
    mean_ic = ic_series.mean()
    ic_std = ic_series.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0.0

    return {
        "mean_ic": round(mean_ic, 4),
        "ic_std": round(ic_std, 4),
        "icir": round(icir, 3),
        "ic_hit_rate_pct": round(100 * (ic_series > 0).mean(), 1),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print backtest metrics to console."""
    print("\n" + "=" * 50)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            print(f"  {label:<30s} {value:>10.3f}")
        else:
            print(f"  {label:<30s} {value:>10}")
    print("=" * 50 + "\n")
