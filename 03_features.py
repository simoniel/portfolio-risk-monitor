"""
03_features.py — Stage 2a: compute features for modelling.

Computes per-day features from raw prices:
  - Daily returns
  - Rolling volatility (annualised)
  - Rolling correlations (SPY vs TLT — the key regime signal)
  - Drawdown from rolling peak

Output: data/processed/features.parquet
Run:    python src/03_features.py
        make features
"""

import pandas as pd
import numpy as np
from utils import load_config, raw_path, processed_path, train_test_split_dates


def load_prices(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(raw_path(cfg, "prices.parquet"))


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def compute_rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised rolling volatility per asset."""
    return returns.rolling(window).std() * np.sqrt(252)


def compute_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    """Drawdown from rolling peak — how far each asset is from its high."""
    rolling_max = prices.cummax()
    return (prices - rolling_max) / rolling_max


def compute_spy_tlt_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """
    Rolling correlation between SPY and TLT.
    This is the classic regime indicator: positive in normal markets,
    turns negative in inflationary regimes (like 2022).
    """
    return returns["SPY"].rolling(window).corr(returns["TLT"]).rename("spy_tlt_corr")


def build_features(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vol_window  = cfg["features"]["vol_window"]
    corr_window = cfg["features"]["corr_window"]

    returns  = compute_returns(prices)
    vol      = compute_rolling_vol(returns, vol_window)
    drawdown = compute_drawdown(prices)
    spy_tlt  = compute_spy_tlt_correlation(returns, corr_window)

    # Rename columns to avoid collisions when combining
    returns.columns  = [f"ret_{c}"  for c in returns.columns]
    vol.columns      = [f"vol_{c}"  for c in vol.columns]
    drawdown.columns = [f"dd_{c}"   for c in drawdown.columns]

    features = pd.concat([returns, vol, drawdown, spy_tlt], axis=1).dropna()

    print(f"Features shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    return features


if __name__ == "__main__":
    cfg    = load_config()
    prices = load_prices(cfg)

    features = build_features(prices, cfg)

    # Confirm no lookahead — rolling windows only use past data
    # (pct_change, rolling().std(), cummax() are all backward-looking)
    print("\n── Train/test split ──")
    train, test = train_test_split_dates(features, cfg)

    # Save
    out = processed_path(cfg, "features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out)
    print(f"\nSaved: {out}")
    print("Next: python src/04_model.py")
