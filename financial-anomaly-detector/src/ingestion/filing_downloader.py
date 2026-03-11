"""
filing_downloader.py — Batch ingestion pipeline.

Builds a multi-company financial dataset from EDGAR XBRL data.

Usage:
    dl = FilingDownloader("data/")
    df = dl.build_dataset(["AAPL", "MSFT", "GOOGL"], years=5)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from ingestion.edgar_client import EDGARClient

logger = logging.getLogger(__name__)

# Representative S&P 500 sample
SP500_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ORCL",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    "WMT", "PG", "KO", "PEP", "MCD",
    "GE", "HON", "CAT", "BA", "MMM",
]


class FilingDownloader:
    """
    Batch downloader — builds a processed financial dataset for multiple
    companies and persists to Parquet/CSV for fast re-loads.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        user_agent: str = "Financial Anomaly Detector research@example.com",
    ):
        self.data_dir  = Path(data_dir)
        self.proc_dir  = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = EDGARClient(user_agent=user_agent)

    def build_dataset(
        self,
        tickers: list,
        form_type: str = "10-K",
        years: int = 5,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Ingest financials for all tickers, cache per-company, return combined DF.
        """
        frames = []
        errors = []

        for ticker in tqdm(tickers, desc="Ingesting SEC filings"):
            # Check per-company cache first
            cache = self.cache_dir / f"{ticker}_{form_type}.parquet"
            if cache.exists():
                try:
                    df = pd.read_parquet(cache)
                    frames.append(df)
                    logger.debug(f"Cache hit: {ticker}")
                    continue
                except Exception:
                    pass  # Corrupted cache — re-fetch

            try:
                df = self.client.get_financial_dataframe(
                    ticker, form_type=form_type
                )
                if df.empty:
                    errors.append((ticker, "No XBRL data returned"))
                    continue

                df["form_type"] = form_type
                df.to_parquet(cache, index=False)
                frames.append(df)
                logger.info(f"✓ {ticker}: {len(df)} rows")

            except Exception as e:
                errors.append((ticker, str(e)))
                logger.error(f"✗ {ticker}: {e}")

        if errors:
            logger.warning(f"Failed tickers ({len(errors)}): {[e[0] for e in errors]}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["end_date"] = pd.to_datetime(combined["end_date"])

        if save:
            out = self.proc_dir / f"financial_dataset_{form_type}.csv"
            combined.to_csv(out, index=False)
            logger.info(f"Saved {len(combined):,} rows → {out}")

        return combined

    def load_cached(self, form_type: str = "10-K") -> pd.DataFrame:
        path = self.proc_dir / f"financial_dataset_{form_type}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No dataset at {path}. Run build_dataset() first.")
        return pd.read_csv(path, parse_dates=["end_date"])

    def load_or_build(
        self,
        tickers: list,
        form_type: str = "10-K",
        years: int = 5,
    ) -> pd.DataFrame:
        try:
            return self.load_cached(form_type)
        except FileNotFoundError:
            return self.build_dataset(tickers, form_type, years)
