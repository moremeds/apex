.PHONY: help install install-dev test lint format clean run-backtest docs sync venv

help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies with uv"
	@echo "  install-dev   Install development dependencies with uv"
	@echo "  sync          Sync dependencies with uv.lock"
	@echo "  venv          Create virtual environment with uv"
	@echo "  test          Run test suite"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  clean         Clean build artifacts"
	@echo "  run-backtest  Run sample backtest"
	@echo "  docs          Build documentation"

venv:
	uv venv

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,ml]"
	uv run pre-commit install

sync:
	uv sync

test:
	uv run pytest tests/ -v --cov=apex

lint:
	uv run ruff check src/
	uv run mypy src/

format:
	uv run black src/ tests/
	uv run ruff check --fix src/

clean:
	rm -rf build/ dist/ *.egg-info .venv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-backtest:
	uv run python scripts/run_backtest.py --strategy ma_cross --symbols AAPL,MSFT

docs:
	uv run mkdocs build

docker-build:
	docker build -t apex:latest -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up