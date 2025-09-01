.PHONY: help install install-dev test lint format clean run-backtest docs

help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run test suite"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  clean         Clean build artifacts"
	@echo "  run-backtest  Run sample backtest"
	@echo "  docs          Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ml]"
	pre-commit install

test:
	pytest tests/ -v --cov=quantum_backtest

lint:
	ruff check src/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-backtest:
	python scripts/run_backtest.py --strategy ma_cross --symbols AAPL,MSFT

docs:
	mkdocs build

docker-build:
	docker build -t quantum-backtest:latest -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up