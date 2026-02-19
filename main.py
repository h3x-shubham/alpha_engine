"""
Alpha Engine — Research Mode Orchestrator

Usage:
    python main.py --config config/base.yaml
    python main.py --config config/base.yaml --csv data.csv
    python main.py --help

Pipeline:
    Config → Universe → Ingest → Clean → Features → Normalize → Labels
    → Walk-Forward Train → Backtest → Print Metrics
"""

from __future__ import annotations

import argparse
import logging
import logging.config
import sys
from pathlib import Path

import yaml

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_loader import ConfigLoader
from core.universe import Universe
from core.types import Cols
from data.ingestion import KiteDataFetcher
from data.cleaning import DataCleaner
from data.storage import PanelStorage
from features.engine import FeatureEngine
from features.normalizer import CrossSectionalNormalizer
from labels.target import TargetBuilder
from models.lgbm_model import LightGBMModel
from models.walk_forward import WalkForwardEngine
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, print_metrics


def setup_logging(config_dir: Path) -> None:
    """Initialize logging from YAML config."""
    log_cfg_path = config_dir / "logging.yaml"
    if log_cfg_path.exists():
        with open(log_cfg_path) as f:
            log_cfg = yaml.safe_load(f)
        # Ensure log directory exists
        Path("logs").mkdir(exist_ok=True)
        logging.config.dictConfig(log_cfg)
    else:
        logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Alpha Engine — Cross-Sectional Stock Return Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config/base.yaml
  python main.py --config config/base.yaml --csv data/ohlcv.csv
  python main.py --config config/base.yaml --cached
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Path to base config directory (default: config/)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file for offline research (bypasses Kite API)",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Use cached data from data_store/ if available",
    )
    return parser.parse_args()


def main() -> None:
    """Main research pipeline orchestrator."""
    args = parse_args()

    # ── 1. Config & Logging ───────────────────
    config_dir = Path(args.config).parent
    setup_logging(config_dir)
    logger = logging.getLogger("alpha_engine.main")
    logger.info("=" * 60)
    logger.info("  ALPHA ENGINE — Research Pipeline")
    logger.info("=" * 60)

    config = ConfigLoader(config_dir)
    config.load("base.yaml", "features.yaml")
    cfg = config.data
    logger.info("Config loaded: %s", list(cfg.keys()))

    # ── 2. Storage ────────────────────────────
    storage = PanelStorage(cfg.get("paths", {}).get("data_store", "data_store"))

    # ── 3. Universe ───────────────────────────
    universe = Universe(cfg)
    logger.info("Universe: %s", universe)

    # ── 4. Data Ingestion ─────────────────────
    if args.cached and storage.exists("cleaned_ohlcv"):
        logger.info("Loading cached cleaned data...")
        ohlcv = storage.load("cleaned_ohlcv")
    elif args.csv:
        logger.info("Loading data from CSV: %s", args.csv)
        ohlcv = KiteDataFetcher.from_csv(args.csv)
    else:
        logger.info("Fetching data from Kite API...")
        fetcher = KiteDataFetcher(
            api_key=cfg.get("broker", {}).get("api_key"),
            access_token=cfg.get("broker", {}).get("access_token"),
            exchange=cfg.get("universe", {}).get("exchange", "NSE"),
        )
        ohlcv = fetcher.fetch_universe(
            tickers=universe.base_tickers,
            start=cfg["dates"]["start"],
            end=cfg["dates"]["end"],
        )
        storage.save(ohlcv, "raw_ohlcv")

    if ohlcv.empty:
        logger.error("No data available. Provide --csv or configure Kite API.")
        sys.exit(1)

    # ── 5. Data Cleaning ──────────────────────
    if not (args.cached and storage.exists("cleaned_ohlcv")):
        cleaner = DataCleaner()
        ohlcv = cleaner.clean(ohlcv)
        storage.save(ohlcv, "cleaned_ohlcv")

    # ── 6. Universe Filtering ─────────────────
    universe.apply_filters(ohlcv)
    ohlcv = universe.filter_panel(ohlcv)
    logger.info("Post-filter panel: %d rows", len(ohlcv))

    # ── 7. Feature Engineering ────────────────
    logger.info("Computing features...")
    feature_engine = FeatureEngine(cfg)
    feature_engine.initialize()
    features = feature_engine.compute(ohlcv)
    logger.info("Features shape: %s", features.shape)

    # ── 8. Cross-Sectional Normalization ──────
    logger.info("Normalizing features cross-sectionally...")
    normalizer = CrossSectionalNormalizer(cfg)
    features_norm = normalizer.normalize(features)

    # ── 9. Label Creation ─────────────────────
    logger.info("Building forward return labels...")
    target_builder = TargetBuilder(cfg)
    labels = target_builder.build(ohlcv)

    # ── 10. Align Features and Labels ─────────
    common_idx = features_norm.index.intersection(labels.index)
    features_aligned = features_norm.loc[common_idx]
    labels_aligned = labels.loc[common_idx]
    logger.info(
        "Aligned dataset: %d samples, %d features",
        len(features_aligned),
        features_aligned.shape[1],
    )

    # ── 11. Walk-Forward Training ─────────────
    logger.info("Starting walk-forward validation...")
    model = LightGBMModel(cfg)
    walk_forward = WalkForwardEngine(cfg)
    oos_predictions = walk_forward.run(model, features_aligned, labels_aligned)

    logger.info(
        "Walk-forward complete: %d OOS predictions, %d folds",
        len(oos_predictions),
        walk_forward.n_folds,
    )

    # ── 12. Backtest ──────────────────────────
    logger.info("Running backtest...")
    backtester = BacktestEngine(cfg)
    bt_results = backtester.run(oos_predictions, ohlcv)

    # ── 13. Metrics ───────────────────────────
    metrics = compute_metrics(
        daily_returns=bt_results["daily_returns"],
        predictions=oos_predictions.iloc[:, 0] if len(oos_predictions.columns) > 0 else None,
        actuals=labels_aligned,
    )
    print_metrics(metrics)

    # ── 14. Feature Importance ────────────────
    try:
        importance = model.feature_importance()
        print("\nTop 10 Feature Importances:")
        print(importance.head(10).to_string())
    except Exception:
        pass

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
