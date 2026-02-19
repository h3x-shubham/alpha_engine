"""
Data ingestion via Zerodha Kite Connect API.

Fetches OHLCV data for the full universe and returns a panel-indexed
DataFrame [date × ticker].
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd

from core.types import Cols, PanelData, make_panel_index

logger = logging.getLogger("alpha_engine.data.ingestion")


class KiteDataFetcher:
    """
    Fetches historical OHLCV data from Zerodha Kite Connect.

    Usage:
        fetcher = KiteDataFetcher(api_key, access_token)
        panel = fetcher.fetch_universe(tickers, start, end)
    """

    INTERVAL_DAY = "day"

    def __init__(
        self,
        api_key: str | None = None,
        access_token: str | None = None,
        exchange: str = "NSE",
        rate_limit_sleep: float = 0.35,
    ) -> None:
        self._api_key = api_key
        self._access_token = access_token
        self._exchange = exchange
        self._rate_limit_sleep = rate_limit_sleep
        self._kite = None
        self._instruments: dict[str, int] = {}

        if api_key and access_token:
            self._connect()

    def _connect(self) -> None:
        """Initialize the Kite Connect client."""
        try:
            from kiteconnect import KiteConnect

            self._kite = KiteConnect(api_key=self._api_key)
            self._kite.set_access_token(self._access_token)
            self._load_instruments()
            logger.info("Kite Connect initialized successfully")
        except ImportError:
            logger.warning(
                "kiteconnect not installed. Using offline/mock mode."
            )
        except Exception as e:
            logger.error("Kite Connect initialization failed: %s", e)

    def _load_instruments(self) -> None:
        """Cache instrument tokens for the configured exchange."""
        if self._kite is None:
            return
        try:
            instruments = self._kite.instruments(self._exchange)
            self._instruments = {
                inst["tradingsymbol"]: inst["instrument_token"]
                for inst in instruments
            }
            logger.info("Loaded %d instruments for %s", len(self._instruments), self._exchange)
        except Exception as e:
            logger.error("Failed to load instruments: %s", e)

    def fetch_ticker(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = INTERVAL_DAY,
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single ticker.

        Returns DataFrame with columns: [date, open, high, low, close, volume].
        """
        if self._kite is None:
            logger.debug("Kite not connected. Returning empty frame for %s", ticker)
            return pd.DataFrame()

        token = self._instruments.get(ticker)
        if token is None:
            logger.warning("Instrument token not found for %s", ticker)
            return pd.DataFrame()

        try:
            records = self._kite.historical_data(
                instrument_token=token,
                from_date=start,
                to_date=end,
                interval=interval,
            )
            df = pd.DataFrame(records)
            if df.empty:
                return df

            df = df.rename(columns={"date": Cols.DATE})
            df[Cols.DATE] = pd.to_datetime(df[Cols.DATE]).dt.normalize()
            df[Cols.TICKER] = ticker

            # Standardize column names
            col_map = {
                "open": Cols.OPEN,
                "high": Cols.HIGH,
                "low": Cols.LOW,
                "close": Cols.CLOSE,
                "volume": Cols.VOLUME,
            }
            df = df.rename(columns=col_map)
            time.sleep(self._rate_limit_sleep)
            return df

        except Exception as e:
            logger.error("Failed to fetch %s: %s", ticker, e)
            return pd.DataFrame()

    def fetch_universe(
        self,
        tickers: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> PanelData:
        """
        Fetch OHLCV data for the full universe and return as PanelData.

        Returns:
            DataFrame with MultiIndex[date, ticker] and OHLCV columns.
        """
        logger.info(
            "Fetching %d tickers from %s to %s", len(tickers), start, end
        )
        frames: list[pd.DataFrame] = []

        for i, ticker in enumerate(tickers):
            logger.debug("Fetching [%d/%d] %s", i + 1, len(tickers), ticker)
            df = self.fetch_ticker(ticker, start, end)
            if not df.empty:
                frames.append(df)

        if not frames:
            logger.warning("No data fetched for any ticker")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.set_index([Cols.DATE, Cols.TICKER]).sort_index()
        logger.info(
            "Fetched panel: %d rows, %d unique dates, %d tickers",
            len(combined),
            combined.index.get_level_values(0).nunique(),
            combined.index.get_level_values(1).nunique(),
        )
        return combined

    @staticmethod
    def from_csv(path: str, date_col: str = "date", ticker_col: str = "ticker") -> PanelData:
        """
        Load OHLCV data from a CSV file (for offline research).

        Expects columns: date, ticker, open, high, low, close, volume.
        """
        df = pd.read_csv(path, parse_dates=[date_col])
        df = df.rename(columns={date_col: Cols.DATE, ticker_col: Cols.TICKER})
        df[Cols.DATE] = pd.to_datetime(df[Cols.DATE]).dt.normalize()
        df = df.set_index([Cols.DATE, Cols.TICKER]).sort_index()
        logger.info("Loaded CSV panel: %d rows from %s", len(df), path)
        return df
