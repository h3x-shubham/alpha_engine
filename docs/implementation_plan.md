# Alpha Engine — Lean Alpha Model (Phase 1)

Focus: **Data → Features → Normalization → Labels → LightGBM → Walk-Forward → Basic Backtest**

---

## Folder Tree

```
D:\programmes\alpha_engine\
├── main.py                        # Research-mode orchestrator
├── requirements.txt
├── setup.py
│
├── config/
│   ├── base.yaml                  # Universe, dates, model params, walk-forward
│   ├── features.yaml              # Feature list & normalization settings
│   └── logging.yaml               # Logging config
│
├── core/
│   ├── __init__.py
│   ├── config_loader.py           # YAML loader → typed dict
│   ├── universe.py                # Multi-asset universe with daily filtering
│   ├── types.py                   # PanelData type alias, enums
│   └── date_utils.py             # Trading calendar helpers
│
├── data/
│   ├── __init__.py
│   ├── ingestion.py               # Kite API fetcher (full universe)
│   ├── cleaning.py                # Fill, winsorize, adjust (cross-sectional)
│   └── storage.py                 # Parquet read/write
│
├── features/
│   ├── __init__.py
│   ├── registry.py                # Decorator-based feature registry
│   ├── engine.py                  # Cross-sectional daily feature processor
│   ├── technical.py               # Momentum, vol, RSI, MACD, etc.
│   ├── microstructure.py          # Volume ratio, Amihud, VWAP deviation
│   ├── normalizer.py             # Per-date z-score / rank normalization
│
├── labels/
│   ├── __init__.py
│   ├── target.py                  # N-day forward return labels
│
├── models/
│   ├── __init__.py
│   ├── base.py                    # BaseModel ABC
│   ├── lgbm_model.py             # LightGBM wrapper
│   └── walk_forward.py           # Walk-forward validation engine
│
├── backtest/
│   ├── __init__.py
│   ├── engine.py                  # Signal-to-PnL replay
│   └── metrics.py                 # Sharpe, IC, drawdown, turnover
│
├── logs/                          # (gitignored)
├── data_store/                    # (gitignored)
│   └── models/
│
└── tests/
    ├── __init__.py
    ├── test_universe.py
    ├── test_features.py
    ├── test_normalizer.py
    ├── test_labels.py
    ├── test_walk_forward.py
    └── test_backtest.py
```

---

## Module Summary

| Module | Key Abstraction | Purpose |
|--------|----------------|---------|
| `core/universe.py` | `Universe` class | Time-varying ticker constituents with liquidity/price filters |
| `core/types.py` | `PanelData` = `MultiIndex[date, ticker]` | Consistent data contract across all modules |
| `features/engine.py` | `FeatureEngine.compute(panel) → panel` | Date-by-date cross-sectional feature computation |
| `features/normalizer.py` | `CrossSectionalNormalizer` | Per-date z-score/rank across universe (no time leakage) |
| `models/walk_forward.py` | `WalkForwardEngine` | Rolling/expanding train windows with embargo gap |
| `backtest/engine.py` | `BacktestEngine` | Replays OOS predictions → simple PnL + metrics |

---

## Execution Flow (Research Mode)

```
python main.py --config config/base.yaml

1. Load config → initialize logger
2. Universe: load tickers, apply filters per date
3. Ingest: fetch OHLCV via Kite API for universe
4. Clean: forward-fill, winsorize, adjust
5. Features: compute cross-sectionally per date
6. Normalize: z-score / rank per date
7. Labels: compute 5-day forward returns
8. Walk-Forward: train LightGBM per window, collect OOS predictions
9. Backtest: replay predictions → PnL, compute metrics
10. Print summary (Sharpe, IC, drawdown)
```

---

## Verification

```bash
cd D:\programmes\alpha_engine
python -m pytest tests/ -v
python main.py --help
python -c "from core import universe, types; print('OK')"
```
