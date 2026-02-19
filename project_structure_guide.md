# Alpha Engine Project Structure & Execution Guide

This document explains the folder structure of the `alpha_engine` repository and provides instructions on how to run the code.

---

## Folder Structure

The project is built using institutional-grade modular architecture. Instead of having one massive script, the code is logically divided into specialized folders based on the pipeline steps.

Here is the breakdown of the folders and what they do:

- **`config/`**: Contains all configuration files (`base.yaml`, `features.yaml`, `logging.yaml`). This is where you set the dates, universe items, credentials, and model hyperparameters.
- **`core/`**: Core infrastructure files. Manages configuration loading, type definitions, date utilities, and stock universe definitions.
- **`data/`**: Handles everything related to fetching data (from Kite API or CSV), cleaning OHLCV data, and saving it to local caching via Parquet format (`storage.py`).
- **`features/`**: Contains the feature engineering engine. This applies technical, microstructure, and custom features to the raw data and handles cross-sectional normalization across your stock universe.
- **`labels/`**: Logic for building target variables, which in this case, are the N-day forward returns we want the model to predict.
- **`models/`**: The Machine Learning implementation. Contains the `LightGBMModel` and the `WalkForwardEngine` for out-of-sample expanding-window cross-validation.
- **`backtest/`**: Simulates trading based on model predictions. Handles the long-short portfolio construction and computes performance metrics (Sharpe ratio, IC, drawdowns).
- **`tests/`**: Contains all unit and integration tests to ensure the pipeline is robust and error-free.
- **`docs/`**: Generated documentation, execution plans, and walkthroughs.

---

## How to Run the Code

**Do NOT run the code folder by folder or file by file.** 

The Python script **`main.py`** situated in the root directory (`alpha_engine/main.py`) acts as the central **Orchestrator**. 

The `main.py` script is dynamically linked to *all* the subfolders (`core`, `data`, `features`, `labels`, `models`, `backtest`). When you execute `main.py`, it imports the classes from these folders automatically and executes the complete pipeline in sequence:

1. Loads Config
2. Creates Universe
3. Ingests & Cleans Data
4. Computes & Normalizes Features
5. Builds Target Labels
6. Performs Walk-Forward Training
7. Runs the Backtest
8. Prints Metrics

### Execution Commands

You only ever need to execute `main.py` from the root of your project directory. 

**1. Standard Run (uses live Kite API data):**
```bash
python main.py --config config/base.yaml
```

**2. Offline Run (uses a local CSV file instead of fetching from API):**
```bash
python main.py --config config/base.yaml --csv data.csv
```

**3. Cached Run (re-uses previously downloaded/cleaned data for faster iterations):**
```bash
python main.py --config config/base.yaml --cached
```

**4. View Help:**
```bash
python main.py --help
```

In summary: **`main.py` does all the heavy lifting by pulling in the code from your subfolders.** You do not need to execute the individual components yourself if you want to run the full pipeline.

---

## Testing Individual Components

While `main.py` is the orchestrator for the full pipeline, during development you will often want to test individual components (like a single data cleaner or feature generator). 

Here are the recommended ways to run the code "folder by folder" or component by component:

### 1. Run Specific Unit Tests

The best way to verify a single component is to run its corresponding tests in the `tests/` folder. For example, to test just the feature engineering engine:
```bash
pytest tests/test_features.py -v
```

### 2. Create a Scratchpad Script

You can create a temporary script in the root directory (e.g., `scratch.py`) to instantiate just the one class you are working on, feed it mock data, and print the output. 
```python
# scratch.py
import pandas as pd
from data.cleaning import DataCleaner

# Load sample data
raw_data = pd.read_parquet("sample_data.parquet")

# Instantiate ONLY the cleaner
cleaner = DataCleaner({"max_missing_pct": 0.1})

# Run and inspect just this component
cleaned_data = cleaner.clean(raw_data)
print(cleaned_data.head())
```

### 3. Use Jupyter Notebooks

For interactive exploration, create a `notebooks/` folder in the root directory, launch Jupyter, and import your components directly to visualize data transformations step-by-step.
