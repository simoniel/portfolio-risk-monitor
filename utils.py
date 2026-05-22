"""
utils.py — shared helpers for all pipeline scripts.
"""

import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Load the central config file. Call this at the top of every script."""
    with open(path) as f:
        return yaml.safe_load(f)


# ── Paths ─────────────────────────────────────────────────────────────────────

def raw_path(cfg: dict, filename: str) -> Path:
    return Path(cfg["data"]["raw_dir"]) / filename

def processed_path(cfg: dict, filename: str) -> Path:
    return Path(cfg["data"]["processed_dir"]) / filename

def plot_path(cfg: dict, filename: str) -> Path:
    return Path(cfg["outputs"]["plots_dir"]) / filename

def report_path(cfg: dict, filename: str) -> Path:
    return Path(cfg["outputs"]["reports_dir"]) / filename

def model_path(cfg: dict, filename: str) -> Path:
    return Path(cfg["outputs"]["models_dir"]) / filename


# ── Date helpers ──────────────────────────────────────────────────────────────

def train_test_split_dates(prices: pd.DataFrame, cfg: dict):
    """
    Split a price/returns DataFrame chronologically.
    Never shuffles — train is always earlier than test.
    """
    train_end  = pd.Timestamp(cfg["portfolio"]["train_end"])
    test_start = pd.Timestamp(cfg["portfolio"]["test_start"])

    train = prices[prices.index <= train_end]
    test  = prices[prices.index >= test_start]

    print(f"Train: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} days)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} days)")

    return train, test


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
