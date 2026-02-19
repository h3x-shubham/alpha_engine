"""
Parquet-based panel data storage.

Caches raw and cleaned data to avoid repeated API calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from core.types import Cols, PanelData

logger = logging.getLogger("alpha_engine.data.storage")


class PanelStorage:
    """
    Read/write PanelData to Parquet files.

    Directory structure:
        data_store/
        ├── raw_ohlcv.parquet
        ├── cleaned_ohlcv.parquet
        ├── features.parquet
        └── models/
    """

    def __init__(self, base_dir: str | Path = "data_store") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, panel: PanelData, name: str) -> Path:
        """
        Save a PanelData DataFrame to Parquet.

        Args:
            panel: DataFrame with MultiIndex[date, ticker].
            name: File name (without extension).
        """
        path = self._base_dir / f"{name}.parquet"
        panel.to_parquet(path, engine="pyarrow")
        logger.info("Saved %s: %d rows → %s", name, len(panel), path)
        return path

    def load(self, name: str) -> PanelData:
        """Load a PanelData DataFrame from Parquet."""
        path = self._base_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        panel = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded %s: %d rows from %s", name, len(panel), path)
        return panel

    def exists(self, name: str) -> bool:
        """Check if a cached data file exists."""
        return (self._base_dir / f"{name}.parquet").exists()

    def list_cached(self) -> list[str]:
        """List all cached data files."""
        return [p.stem for p in self._base_dir.glob("*.parquet")]
