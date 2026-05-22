.PHONY: all ingest eda features model rag summarise evaluate clean help

help:
	@echo "Portfolio Risk Monitor — available targets:"
	@echo "  make ingest     Download price data (Stage 1)"
	@echo "  make eda        Run exploratory analysis (Stage 1)"
	@echo "  make features   Compute returns, vol, drawdown (Stage 2)"
	@echo "  make model      Regime detection + anomaly (Stage 2)"
	@echo "  make rag        Build news vector store (Stage 3)"
	@echo "  make summarise  Generate LLM risk report (Stage 4)"
	@echo "  make evaluate   Run eval harness (Stage 5)"
	@echo "  make all        Run full pipeline"
	@echo "  make clean      Remove processed data and outputs"

ingest:
	python src/01_ingest.py

eda: ingest
	python src/02_eda.py

features: eda
	python src/03_features.py

model: features
	python src/04_model.py

rag: model
	python src/05_rag.py

summarise: rag
	python src/06_summarise.py

evaluate: summarise
	python src/07_evaluate.py

all: evaluate

clean:
	rm -rf data/processed/*
	rm -rf outputs/plots/*
	rm -rf outputs/reports/*
	rm -rf outputs/models/*
	@echo "Cleaned. Raw data preserved."
