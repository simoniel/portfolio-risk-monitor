"""
02_eda.py — Stage 1b: exploratory data analysis.

Answers the same questions as the Parkinson's EDA:
  - Data quality (missing dates, gaps)
  - Return distributions (fat tails, outliers)
  - Correlation structure
  - Rolling volatility — where are the regimes?

Outputs: outputs/plots/eda_*.png
Run:     python src/02_eda.py
         make eda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from utils import load_config, raw_path, plot_path, save_fig


def load_prices(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(raw_path(cfg, "prices.parquet"))


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# ── Plot 1: Normalised price growth ──────────────────────────────────────────

def plot_normalised_prices(prices: pd.DataFrame, cfg: dict) -> None:
    """Base-100 chart — the executive summary of the dataset."""
    normed = prices / prices.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in normed.columns:
        ax.plot(normed.index, normed[col], label=col, linewidth=1.2)

    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Normalised price growth (base 100)", fontsize=13)
    ax.set_ylabel("Index (100 = start)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()

    save_fig(fig, plot_path(cfg, "eda_01_normalised_prices.png"))


# ── Plot 2: Return distributions ─────────────────────────────────────────────

def plot_return_distributions(returns: pd.DataFrame, cfg: dict) -> None:
    n = len(returns.columns)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=False)
    axes = axes.flatten()

    for i, col in enumerate(returns.columns):
        axes[i].hist(returns[col], bins=60, edgecolor="none", alpha=0.8)
        axes[i].set_title(col, fontsize=9)
        axes[i].axvline(0, color="red", linewidth=0.8, linestyle="--")

    fig.suptitle("Daily return distributions", fontsize=12)
    fig.tight_layout()
    save_fig(fig, plot_path(cfg, "eda_02_return_distributions.png"))


# ── Plot 3: Correlation heatmap ───────────────────────────────────────────────

def plot_correlation(returns: pd.DataFrame, cfg: dict) -> None:
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1,
        linewidths=0.3, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Asset correlation matrix (full period)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, plot_path(cfg, "eda_03_correlation_heatmap.png"))


# ── Plot 4: Rolling volatility of SPY ────────────────────────────────────────

def plot_rolling_vol(returns: pd.DataFrame, cfg: dict) -> None:
    vol_window = cfg["features"]["vol_window"]
    spy_vol = returns["SPY"].rolling(vol_window).std() * np.sqrt(252)  # annualised

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(spy_vol.index, spy_vol, alpha=0.4, color="steelblue")
    ax.plot(spy_vol.index, spy_vol, linewidth=0.8, color="steelblue")
    ax.set_title(f"SPY rolling {vol_window}-day annualised volatility", fontsize=12)
    ax.set_ylabel("Annualised vol")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, plot_path(cfg, "eda_04_spy_rolling_vol.png"))


# ── Outlier scan — equivalent to the MDVP:Fhi(Hz) check ─────────────────────

def outlier_scan(returns: pd.DataFrame, threshold_std: float = 3.5) -> None:
    print(f"\n── Outlier scan (>{threshold_std} std from mean) ──")
    outlier_counts = {}

    for col in returns.columns:
        mean, std = returns[col].mean(), returns[col].std()
        hi = returns[col][returns[col].abs() > mean + threshold_std * std]
        if not hi.empty:
            print(f"{col}: {len(hi)} outlier days")
            for date, val in hi.items():
                print(f"   {date.date()}  {val:+.2%}")
            outlier_counts[col] = len(hi)

    print(f"\nMost volatile assets by outlier day count:")
    for asset, count in sorted(outlier_counts.items(), key=lambda x: -x[1]):
        print(f"  {asset}: {count} days")


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(returns: pd.DataFrame) -> None:
    print("\n── Return summary (annualised) ──")
    summary = pd.DataFrame({
        "Mean return":  returns.mean() * 252,
        "Volatility":   returns.std() * np.sqrt(252),
        "Sharpe (0rf)": (returns.mean() / returns.std()) * np.sqrt(252),
        "Min day":      returns.min(),
        "Max day":      returns.max(),
    }).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    cfg     = load_config()
    prices  = load_prices(cfg)
    returns = compute_returns(prices)

    print(f"Loaded: {prices.shape[0]} days × {prices.shape[1]} assets")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    print_summary(returns)
    outlier_scan(returns)

    print("\nGenerating plots...")
    plot_normalised_prices(prices, cfg)
    plot_return_distributions(returns, cfg)
    plot_correlation(returns, cfg)
    plot_rolling_vol(returns, cfg)

    print("\nStage 1b complete. Plots saved to outputs/plots/")
    print("Next: python src/03_features.py")
