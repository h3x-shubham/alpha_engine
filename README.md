<p align="center">
  <h1 align="center">⚡ Alpha Engine</h1>
  <p align="center">
    <strong>Quant-grade ML system for cross-sectional stock return prediction</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#usage">Usage</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#testing">Testing</a>
  </p>
</p>

---

## Overview

Alpha Engine is a **production-ready research pipeline** for predicting 5-day forward stock returns using cross-sectional machine learning. Built with institutional-grade standards, it provides a modular framework covering the full alpha research lifecycle — from data ingestion to walk-forward validation and backtesting.

## Features

- 🔬 **Cross-Sectional Feature Engineering** — Technical, microstructure, and custom features with a pluggable registry
- 📊 **Cross-Sectional Normalization** — Rank-based and z-score normalization across the stock universe
- 🌲 **LightGBM Modeling** — Gradient-boosted trees optimized for tabular financial data
- 🔄 **Walk-Forward Validation** — Expanding-window out-of-sample evaluation with configurable folds
- 📈 **Backtesting Engine** — Long-short portfolio simulation with realistic transaction cost modeling
- 🔗 **Kite API Integration** — Native support for Zerodha Kite data (with CSV fallback for offline research)
- 💾 **Panel Storage** — Parquet-based caching for fast data reload

## Architecture

```
alpha_engine/
│
├── main.py                  # Pipeline orchestrator
├── config/
│   ├── base.yaml            # Core configuration (universe, dates, model params)
│   ├── features.yaml        # Feature definitions & parameters
│   └── logging.yaml         # Logging configuration
│
├── core/                    # Core infrastructure
│   ├── config_loader.py     # YAML config management
│   ├── types.py             # Column name constants & type definitions
│   ├── universe.py          # Stock universe & liquidity filtering
│   └── date_utils.py        # Trading calendar utilities
│
├── data/                    # Data layer
│   ├── ingestion.py         # Kite API fetcher + CSV loader
│   ├── cleaning.py          # OHLCV cleaning & validation
│   └── storage.py           # Parquet panel storage
│
├── features/                # Feature engineering
│   ├── engine.py            # Feature computation orchestrator
│   ├── technical.py         # Technical indicators (momentum, volatility, etc.)
│   ├── microstructure.py    # Market microstructure features
│   ├── normalizer.py        # Cross-sectional normalization
│   └── registry.py          # Pluggable feature registry
│
├── labels/                  # Target variable
│   └── target.py            # Forward return label builder
│
├── models/                  # ML models
│   ├── base.py              # Abstract model interface
│   ├── lgbm_model.py        # LightGBM implementation
│   └── walk_forward.py      # Walk-forward validation engine
│
├── backtest/                # Backtesting
│   ├── engine.py            # Long-short portfolio simulator
│   └── metrics.py           # Performance metrics (Sharpe, IC, drawdown, etc.)
│
└── tests/                   # Unit & integration tests
```

## Quickstart

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/h3x-shubham/alpha_engine.git
cd alpha_engine

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Usage

### Pipeline

The full pipeline flows through these stages:

```
Config → Universe → Ingest → Clean → Features → Normalize → Labels
→ Walk-Forward Train → Backtest → Print Metrics
```

### Run with CSV data (offline research)

```bash
python main.py --config config/base.yaml --csv path/to/ohlcv.csv
```

### Run with Kite API (live data)

```bash
python main.py --config config/base.yaml
```

### Run with cached data

```bash
python main.py --config config/base.yaml --cached
```

## Configuration

All pipeline behaviour is controlled via YAML configs in the `config/` directory:

| File | Purpose |
|---|---|
| `base.yaml` | Universe, date range, model hyperparameters, broker credentials |
| `features.yaml` | Feature definitions, lookback windows, normalization settings |
| `logging.yaml` | Log levels, file handlers, formatting |

## Key Dependencies

| Package | Purpose |
|---|---|
| `lightgbm` | Gradient boosted tree model |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | ML utilities & metrics |
| `shap` | Feature importance & explainability |
| `kiteconnect` | Zerodha broker API |
| `pyarrow` | Parquet storage |

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## License

This project is private. All rights reserved.

---

<p align="center">
  Built for institutional-grade alpha research 🧠
</p>
