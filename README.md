# Portfolio Risk Monitor

A five-stage pipeline that combines classical ML with LLM-powered narrative summaries
to monitor portfolio risk — detecting market regimes, flagging anomalous days,
and generating plain-language risk reports grounded in financial news.

## Project stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1a | `01_ingest.py` | Download adjusted prices via yfinance |
| 1b | `02_eda.py` | Exploratory analysis: distributions, correlations, volatility |
| 2a | `03_features.py` | Compute returns, rolling vol, drawdown, SPY/TLT correlation |
| 2b | `04_model.py` | Regime detection (KMeans) + anomaly scoring (IsolationForest) |
| 3  | `05_rag.py` | Embed news headlines, build ChromaDB vector store |
| 4  | `06_summarise.py` | Generate LLM risk summaries via Anthropic API or Ollama |
| 5  | `07_evaluate.py` | Eval harness: grounding, hallucination check, backtesting |

## Setup

```bash
# Clone and enter
git clone <your-repo-url>
cd portfolio-risk-monitor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
```

## Configuration

All settings live in `config.yaml` — tickers, date range, model parameters, LLM provider.
Edit this file rather than touching the scripts directly.

## Running

```bash
# Run individual stages
make ingest       # Stage 1a: download data
make eda          # Stage 1b: EDA plots
make features     # Stage 2a: feature engineering

# Or run everything
make all

# See all targets
make help
```

## Data sources

- **Prices**: [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance (free, no registration)
- **News**: TBD in Stage 3 — Yahoo Finance RSS / FRED economic releases
- **Period**: 2019-01-01 to 2024-12-31 (covers COVID crash, 2022 rate shock, bull market)

## Portfolio

10 assets chosen for regime diversity:

`SPY QQQ IWM EFA EEM TLT GLD VNQ USO BTC-USD`

## Key design decisions

- **Chronological train/test split** — train on 2019–2022, test on 2023–2024. Never random shuffle.
- **No lookahead in features** — all rolling windows are backward-looking.
- **Config-driven** — tickers, windows, model params all in `config.yaml`, not hardcoded.
- **Raw data is read-only** — `data/raw/` is never modified after download.

## Results

*To be filled in as stages are completed.*

---

## Status

- [x] Stage 1a: data ingestion
- [x] Stage 1b: EDA
- [ ] Stage 2a: feature engineering
- [ ] Stage 2b: regime modelling
- [ ] Stage 3: RAG layer
- [ ] Stage 4: LLM summaries
- [ ] Stage 5: evaluation
