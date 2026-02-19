# Alpha Engine — Walkthrough

## What Was Built

A lean, production-grade ML system for **cross-sectional stock return prediction** at `D:\programmes\alpha_engine`.

---

## Folder Tree (46 files)

```
alpha_engine/
├── main.py                        # Research pipeline orchestrator
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── .gitignore
│
├── config/
│   ├── base.yaml                  # Universe, dates, model, walk-forward, backtest
│   ├── features.yaml              # Feature list & normalization settings
│   └── logging.yaml               # Rotating file + console logging
│
├── core/
│   ├── config_loader.py           # YAML → typed dict with env var expansion
│   ├── universe.py                # Multi-asset universe with daily filtering
│   ├── types.py                   # PanelData = MultiIndex[date, ticker]
│   └── date_utils.py             # NSE trading calendar + walk-forward splits
│
├── data/
│   ├── ingestion.py               # Kite API fetcher + CSV fallback
│   ├── cleaning.py                # Dedup, fill, winsorize, volume fix
│   └── storage.py                 # Parquet cache
│
├── features/
│   ├── registry.py                # @feature decorator + singleton registry
│   ├── engine.py                  # Cross-sectional daily feature processor
│   ├── technical.py               # 11 indicators (momentum, vol, RSI, MACD...)
│   ├── microstructure.py          # 5 indicators (volume ratio, Amihud, VWAP...)
│   └── normalizer.py             # Per-date z-score / rank / minmax
│
├── labels/
│   └── target.py                  # N-day forward returns (log/simple)
│
├── models/
│   ├── base.py                    # BaseModel ABC
│   ├── lgbm_model.py             # LightGBM wrapper with early stopping
│   ├── walk_forward.py           # Rolling/expanding train with embargo
│
├── backtest/
│   ├── engine.py                  # Top-N equal-weight signal replay
│   └── metrics.py                 # Sharpe, Sortino, IC, drawdown, Calmar
│
└── tests/                         # 6 test files covering all core modules
```

---

## Key Design Patterns

### 1. PanelData Convention
All DataFrames use `MultiIndex[date, ticker]`. This enables `groupby(level='date')` for cross-sectional operations — the same pattern used in live production.

### 2. Universe Abstraction
[universe.py](file:///D:/programmes/alpha_engine/core/universe.py) provides `get_tickers(date)` — a time-varying view of the investable universe. All downstream modules receive dynamic ticker sets, not hardcoded lists.

### 3. Cross-Sectional Processing
The [feature engine](file:///D:/programmes/alpha_engine/features/engine.py) computes features per-ticker then the [normalizer](file:///D:/programmes/alpha_engine/features/normalizer.py) z-scores per-date across the universe. No time-series information leaks into normalization.

### 4. Self-Registering Features
The `@feature` decorator in [registry.py](file:///D:/programmes/alpha_engine/features/registry.py) auto-registers features. To add a new feature, just decorate a function — no wiring needed.

---

## How to Run

```bash
# Install dependencies
cd D:\programmes\alpha_engine
pip install -r requirements.txt

# Run with CSV data (offline research)
python main.py --csv path/to/ohlcv.csv

# Run with Kite API (set env vars first)
set KITE_API_KEY=your_key
set KITE_ACCESS_TOKEN=your_token
python main.py --config config/base.yaml

# Re-run with cached data
python main.py --cached

# Run tests
python -m pytest tests/ -v
```

---

## CSV Format Expected

If using `--csv`, the file should have these columns:

| date | ticker | open | high | low | close | volume |
|------|--------|------|------|-----|-------|--------|
| 2024-01-02 | RELIANCE | 2500 | 2520 | 2480 | 2510 | 5000000 |

---

## Next Steps (Phase 2)

When ready to go beyond alpha research:
- Portfolio construction & mean-variance optimization
- Risk management (exposure limits, drawdown circuit breaker)
- Transaction cost modeling (Zerodha fee schedule, slippage)
- Live execution via Kite API
- Experiment tracking & HTML reports
