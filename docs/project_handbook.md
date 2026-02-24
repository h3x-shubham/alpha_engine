# Alpha Engine Project Handbook: Institutional Quant Grade 📖

Welcome to the Alpha Engine Handbook. This document serves as a comprehensive, institutional-grade reference guide for developing quantitative trading infrastructure. It breaks down every folder, module, and mathematical concept utilized in this repository.

This handbook is designed for Quantitative Researchers and Developers to ensure architectural consistency, rigorous out-of-sample validation, and scalable feature engineering.

---

## 🏢 1. Configuration (`config/`)

In quantitative development, separating pipeline logic from exogenous parameters is non-negotiable. Configuration files prevent hardcoding, enabling rapid experimentation through grid searches and external orchestrators (like Airflow).

### 📄 `base.yaml`
**Purpose:** Sets the global state for the trading environment, including simulation horizons, universe definitions, paths, and core model hyperparameters.
**Subtopics:**
*   **Environment Setup:** Defining absolute/relative paths to the Data Lake (e.g., local Parquet files, AWS S3 buckets).
*   **Temporal Horizons:** Tightly defining the `in_sample` (training) and `out_of_sample` (testing/live) periods to strictly prevent data leakage.
*   **Execution Assumptions:** Latencies, assumed slip, and benchmark indices.

*Quant-Grade Example:*
```yaml
# config/base.yaml
environment:
  data_lake: "s3://quant-data/india/nifty50/"
  local_cache: "data/cache/"
temporal_parameters:
  train_start: "2015-01-01"
  train_end: "2020-12-31"
  test_start: "2021-01-01"   # Strict OOS boundary
universe:
  type: "dynamic_liquidity"
  base_index: "NIFTY50"
  min_adv_usd: 5000000      # Minimum Average Daily Volume (USD) filter
execution:
  assumed_slippage_bps: 5.0 # 5 basis points slip per trade
  commission_bps: 2.0
```

### 📄 `features.yaml`
**Purpose:** A declarative registry for the feature generation pipeline. It specifies the mathematical indicators, lookback windows, and cross-sectional transformations to be applied to the universe.
**Subtopics:**
*   **Multi-timeframe Analysis:** Generating features across different granularities (e.g., 5-min, hourly, daily) using moving windows.
*   **Feature Taxonomy:** Splitting indicators logically into momentum, mean-reversion, volatility, and microstructure groups.

*Quant-Grade Example:*
```yaml
# config/features.yaml
momentum:
  exponential_moving_average:
    spans: [10, 21, 55, 200]
    normalize_type: "cross_sectional_zscore"
volatility:
  atr_normalized:
    period: 14
microstructure:
  roll_measure_spread:
    window: 20
    clip_outliers: true # Winsorization at 1st/99th percentiles
```

### 📄 `logging.yaml`
**Purpose:** Asynchronous, multi-sink logging setup.
**Subtopics:** Critical for post-trade forensics. Differentiating between `INFO` (pipeline heartbeat) and `DEBUG` (vectorized operation times, NaN counts in matrices).

---

## ⚙️ 2. Core Infrastructure (`core/`)

This module houses the universally shared utilities that enforce type safety, handle the idiosyncrasies of financial calendars, and maintain universe states.

### 📄 `types.py`
**Purpose:** Enforces strict typing using Python's `typing` module, preventing runtime crashes during computationally expensive vectorized operations.
**Subtopics:**
*   **Strong Typing:** Avoiding ambiguous `kwargs`.
*   **Enums for State Machines:** Representing execution states explicitly.

*Quant-Grade Example:*
```python
# core/types.py
from typing import Dict, List, Union
from enum import Enum, auto
import pandas as pd
import numpy as np

class SignalDirection(Enum):
    LONG = 1
    FLAT = 0
    SHORT = -1

# Type Aliases for code readability
FeaturesMatrix = pd.DataFrame   # Shape: (Dates x Tickers, Features)
TargetVector = pd.Series        # Shape: (Dates x Tickers,)
Predictions = np.ndarray
```

### 📄 `config_loader.py`
**Purpose:** Robustly loads YAMLs, merges them with environment variables, and validates schemas (e.g., using `pydantic`).

### 📄 `date_utils.py`
**Purpose:** Resolves the massive headache of market calendars, holidays, and lookahead biases in time manipulation.
**Subtopics:**
*   **Holiday Calendars:** Utilizing `pandas_market_calendars` or custom NSE/BSE holiday lists.
*   **Alignment:** Ensuring that the Close of T-1 is always aligned correctly with the Open of T without peeking.

*Quant-Grade Example:*
```python
# core/date_utils.py
import pandas_market_calendars as mcal
from datetime import date, timedelta
import pandas as pd

def get_trading_days(start: str, end: str, exchange: str = 'NSE') -> pd.DatetimeIndex:
    """Returns an array of actual trading days, accounting for exchange holidays."""
    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=start, end_date=end)
    return schedule.index
```

### 📄 `universe.py`
**Purpose:** Manages the survivor-bias-free tracking of tradable assets.
**Subtopics:**
*   **Survivorship Bias:** A critical quantitative error where delisted stocks are ignored. The universe module must retain historical constituents of an index (e.g., Yes Bank in Nifty 50 during 2018).
*   **Dynamic Filtering:** Dropping illiquid or suspended symbols systematically before feature generation.

---

## 💾 3. Data Pipeline (`data/`)

The ETL (Extract, Transform, Load) backbone. Garbage in, garbage out out—this module guarantees data fidelity.

### 📄 `ingestion.py`
**Purpose:** Interfaces with data vendors (Kite Connect, Bloomberg, Polygon) orchestrating the download of massive historical tick or OHLCV datasets.
**Subtopics:**
*   **Pagination & Rate Limiting:** Respecting API constraints.
*   **Idempotency:** Designing fetchers that can fail halfway and resume without duplicating rows.

### 📄 `cleaning.py`
**Purpose:** The single most important part of the ETL. Ingests raw prints and handles structural breaks.
**Subtopics:**
*   **Corporate Actions Adjustments:** Adjusting the historical time series for Stock Splits, Dividends, and Mergers to prevent artificial price gaps.
*   **Forward-Filling (ffill):** Handling sparse trading days (e.g., trading halts) without introducing NaNs, while strictly limiting the ffill duration to avoid trading stale data.
*   **Bad Tick Filtering:** Using MAD (Median Absolute Deviation) to detect and remove impossible spikes.

*Quant-Grade Example:*
```python
# data/cleaning.py
import numpy as np
import pandas as pd

class DataCleaner:
    def handle_outliers_mad(self, series: pd.Series, threshold: float = 5.0) -> pd.Series:
        """Removes impossible price spikes using Median Absolute Deviation."""
        median = series.median()
        mad = np.abs(series - median).median()
        upper_bound = median + (threshold * mad)
        lower_bound = median - (threshold * mad)
        
        # Clip or replace with NaN
        return series.clip(lower=lower_bound, upper=upper_bound)
        
    def adjust_for_splits(self, df: pd.DataFrame, splits_df: pd.DataFrame) -> pd.DataFrame:
        """Mathematical adjustment of OHLC based on cumulative split factors."""
        # Implementation relying on cumulative products of split ratios
        pass
```

### 📄 `storage.py`
**Purpose:** Highly optimized I/O operations for disk caching.
**Subtopics:**
*   **Columnar Formats:** Utilizing Apache Parquet for 10x compression over CSVs and blazing-fast selective column reads (e.g., reading just the `Close` price across 500 stocks without loading `Volume`).

---

## 🧬 4. Feature Engineering (`features/`)

Converts the sanitized DataFrame into a highly-dimensional matrix of predictive variables (Alphas). Vectorization is absolutely mandatory here; `for` loops are strictly banned due to performance bottlenecks.

### 📄 `technical.py` & `microstructure.py`
**Purpose:** The mathematical implementation of alpha factors.
**Subtopics:**
*   **Technical:** MACD, RSI, Bollinger Bands—classic metrics mapped to momentum and mean reversion.
*   **Microstructure:** Advanced metrics analyzing the bid-ask spread and volume profiles using only OHLCV data. 
    *   *Corwin-Schultz Spread Estimator*
    *   *Amihud Illiquidity*
    *   *Roll Measure*

*Quant-Grade Example (Roll Measure):*
```python
# features/microstructure.py
import pandas as pd
import numpy as np

def calculate_roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Estimates the effective bid-ask spread from the serial covariance of price changes.
    Used to detect illiquidity and potential mean-reversion conditions.
    """
    delta_p = close_prices.diff()
    # Covariance between delta_p_t and delta_p_{t-1}
    covar = delta_p.rolling(window).cov(delta_p.shift(1))
    
    # Spread formula: 2 * sqrt(-Cov) if Cov < 0, else 0
    effective_spread = np.where(covar < 0, 2 * np.sqrt(-covar), 0)
    return pd.Series(effective_spread, index=close_prices.index)
```

### 📄 `normalizer.py`
**Purpose:** Ensures features are stationary and comparable across completely different assets.
**Subtopics:**
*   **Cross-Sectional Z-Scoring:** For each timestamp, subtract the mean of the *cross-section* (all stocks on that day) and divide by the cross-sectional standard deviation. This neutralizes market-wide effects (Market Beta).
*   **Winsorization:** Capping fat tails (extreme outliers) at the 1st and 99th percentiles so they don't skew the Machine Learning model's gradients.

---

## 🎯 5. Labels (`labels/`)

Defining the Ground Truth. What is the model mathematically optimizing for?

### 📄 `target.py`
**Purpose:** Generates the highly specific values the ML model must learn to predict from the features.
**Subtopics:**
*   **Forward Returns ($Y_{t+n}$):** Calculating the percentage change from $Open_{t+1}$ to $Close_{t+n}$. *Crucially*, we execute trades dynamically, meaning we cannot use $Close_t$ as the entry price for a signal generated right at $Close_t$.
*   **Risk-Adjusted Targets:** Designing targets like Volatility-Scaled Returns ($Returns / \sigma$) to penalize predicting gains in highly erratic stocks.

*Quant-Grade Example:*
```python
# labels/target.py
import pandas as pd

def calculate_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Calculates the 5-day forward return.
    Assumes signal generated at Close(T). 
    Execution happens at Open(T+1). 
    Position closed at Close(T+horizon).
    """
    # Shift prices backwards to align future prices with current row
    entry_price = df['Open'].shift(-1)
    exit_price = df['Close'].shift(-horizon)
    
    forward_return = (exit_price - entry_price) / entry_price
    return forward_return
```

---

## 🧠 6. Machine Learning Models (`models/`)

The core alpha generation engine mapping $X$ (Features) to $y$ (Targets).

### 📄 `lgbm_model.py`
**Purpose:** Wraps the native `lightgbm` C++ API. Gradient boosted trees are highly robust to multicollinearity and non-linear interactions common in financial metrics.
**Subtopics:**
*   **Objective Functions:** Custom asymmetric loss functions (e.g., penalizing false positives—buying a crashing stock—heavier than false negatives).
*   **Early Stopping:** Preventing overfitting by stopping tree growth when out-of-sample validation error starts rising.

### 📄 `walk_forward.py`
**Purpose:** The institutional gold-standard for cross-validation in time series.
**Subtopics:**
*   **Expanding Window / Purged K-Fold:** You cannot use standard K-Fold CV. If you train on 2020 and 2022 to predict 2021, you have introduced lookahead bias.
*   **Purging / Embargoing:** Leaving a gap of $N$ days between the Train set and the Validation set to ensure the $N$-day forward returns do not overlap logic, violating independence.

---

## 📈 7. Backtesting (`backtest/`)

The simulation engine determining if the predictive alpha survives the friction of reality.

### 📄 `engine.py`
**Purpose:** Converts abstract predictions (e.g., a float array `[0.05, -0.02, 0.01]`) into tangible portfolio weights and capital allocation.
**Subtopics:**
*   **Signal Discretization:** e.g., "Go Long the top decile (top 10%), Short the bottom decile, flat everything else."
*   **Capital Allocation:** Equal weight vs. Volatility Parity weighting.
*   **Slippage & Impact:** Modeling execution friction. Applying a 5 bps penalty to every transaction turnover.

### 📄 `metrics.py`
**Purpose:** Calculates the hard institutional KPIs required to pitch a strategy.
**Subtopics:**
*   **Information Coefficient (IC):** Spearman Rank Correlation between predicted returns and actual returns. (IC > 0.05 is exceptional).
*   **Annualized Sharpe & Sortino Ratios:** Return per unit of total risk vs. downside risk.
*   **Turnover & Capacity:** Measuring how frequently the portfolio churns. High turnover strategies die to trading fees.

---

## 🎬 8. Execution & Continuous Integration (`main.py` & `tests/`)

### 📄 `main.py`
**Purpose:** The Directed Acyclic Graph (DAG) executor. It cleanly injects dependencies: Config $\rightarrow$ Data $\rightarrow$ Engine $\rightarrow$ Backtest $\rightarrow$ Tearsheet.

### 📂 `tests/`
**Purpose:** If code is not tested, it is broken. Period.
**Subtopics:**
*   **Determinism:** ML seeds must be fixed. Running `main.py` twice must yield the exact same Sharpe calculation to the 4th decimal point.
*   **Mocking:** `test_ingestion.py` should mock API responses from Kite instead of making live network calls to ensure tests run offline in milliseconds.
