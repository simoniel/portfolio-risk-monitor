"""
01_ingest.py — Stage 1a: download adjusted closing prices via yfinance.

Output: data/raw/prices.parquet
Run:    python src/01_ingest.py
        make ingest
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from utils import load_config, raw_path


def download_prices(cfg: dict) -> pd.DataFrame:
    tickers    = cfg["portfolio"]["tickers"]
    start_date = cfg["portfolio"]["start_date"]
    end_date   = cfg["portfolio"]["end_date"]

    print(f"Downloading {len(tickers)} tickers: {start_date} → {end_date}")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=True)
    prices = raw["Close"]

    print(f"\nShape: {prices.shape[0]} trading days × {prices.shape[1]} assets")
    return prices


def check_quality(prices: pd.DataFrame) -> None:
    """Basic quality checks — mirrors the EDA checks from the Parkinson's project."""
    print("\n── Missing values per asset ──")
    missing = prices.isnull().sum()
    print(missing[missing > 0] if missing.any() else "None — clean dataset")

    print("\n── Date range ──")
    print(f"First: {prices.index[0].date()}")
    print(f"Last:  {prices.index[-1].date()}")

    print("\n── Price ranges (first/last row) ──")
    print(prices.iloc[[0, -1]].T.rename(columns={prices.index[0]: "first", prices.index[-1]: "last"}))


def save(prices: pd.DataFrame, cfg: dict) -> None:
    out = raw_path(cfg, "prices.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    cfg    = load_config()
    prices = download_prices(cfg)
    check_quality(prices)
    save(prices, cfg)
    print("\nStage 1a complete. Next: python src/02_eda.py")
