# APEX Development Makefile
# Quick commands for common development tasks

.PHONY: install run run-dev run-prod run-demo run-headless lint format type-check dead-code complexity quality test test-all coverage clean help validate-fast strategy-compare strategy-verify strategy-compare-quick pead pead-test pead-screen momentum momentum-update momentum-backtest momentum-test r2-universe r2-backfill r2-backfill-test r2-delta r2-validate r2-market-caps server-dev server web-install web-dev web-build live tunnel jobs-momentum jobs-pead jobs-strategy-compare

# Virtual environment - use .venv/bin executables directly
VENV := .venv/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip

# Colors for terminal output
BOLD := $(shell tput bold)
RESET := $(shell tput sgr0)
GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)

help:
	@echo "$(BOLD)APEX Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup:$(RESET)"
	@echo "  make install        Install all dependencies with uv"
	@echo ""
	@echo "$(GREEN)Run:$(RESET)"
	@echo "  make run            Start TUI dashboard (dev mode, verbose)"
	@echo "  make run-prod       Start TUI dashboard (production mode)"
	@echo "  make run-demo       Start TUI dashboard (demo/offline mode)"
	@echo "  make run-headless   Run without TUI (headless mode)"
	@echo ""
	@echo "$(GREEN)Quality:$(RESET)"
	@echo "  make lint           Check formatting (black, isort, flake8)"
	@echo "  make format         Auto-fix formatting (black, isort)"
	@echo "  make type-check     Run mypy type checker"
	@echo "  make quality        Run all quality checks"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  make test           Run unit tests"
	@echo "  make test-all       Run all tests (unit + integration)"
	@echo "  make coverage       Run tests with HTML coverage report"
	@echo ""
	@echo "$(GREEN)Validation:$(RESET)"
	@echo "  make validate-fast  PR gate (fast, 10 symbols)"
	@echo "  make strategy-compare  Run strategy backtests + comparison"
	@echo "  make strategy-compare-quick  Quick compare (3 symbols, 1yr)"
	@echo "  make strategy-verify   Full strategy verification (tests + compare + lint)"
	@echo ""
	@echo "$(GREEN)PEAD Screener:$(RESET)"
	@echo "  make pead           Full pipeline (earnings + attention + screen + track + stats)"
	@echo "  make pead-test      Run PEAD unit tests"
	@echo "  make pead-screen    Screen from cache (still tracks + resolves by default)"
	@echo ""
	@echo "$(GREEN)Momentum Screener:$(RESET)"
	@echo "  make momentum                     Refresh + screen"
	@echo "  make quantitative-moment          Alias for momentum"
	@echo "  make quantitative-moment-backtest Walk-forward backtest + ablation"
	@echo "  make quantitative-moment-test     Run unit tests"
	@echo ""
	@echo "$(GREEN)R2 Data Pipeline:$(RESET)"
	@echo "  make r2-universe       Screen universe (FMP → filter → ~500 symbols → R2)"
	@echo "  make r2-backfill       Full backfill (all symbols, 2019-present)"
	@echo "  make r2-backfill-test  Quick test (5 symbols)"
	@echo "  make r2-delta          Incremental delta update"
	@echo "  make r2-validate       Generate data_quality.json only"
	@echo "  make r2-market-caps    Update market caps → R2"
	@echo ""
	@echo "$(GREEN)Compute Jobs (API triggers):$(RESET)"
	@echo "  make jobs-momentum          Run momentum screener"
	@echo "  make jobs-pead              Run PEAD screener"
	@echo "  make jobs-strategy-compare  Run strategy comparison backtest"
	@echo ""
	@echo "$(GREEN)Live Dashboard:$(RESET)"
	@echo "  make server-dev        Start FastAPI server (dev, auto-reload, :8080)"
	@echo "  make server            Start FastAPI server (production, :8080)"
	@echo "  make web-dev           Start React dev server (:5173)"
	@echo "  make web-build         Build React frontend for production"
	@echo "  make tunnel            Start Cloudflare Tunnel"
	@echo ""
	@echo "$(GREEN)Other:$(RESET)"
	@echo "  make clean          Remove build artifacts"

# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

install:
	@echo "$(BOLD)Installing dependencies with uv...$(RESET)"
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev,observability,server,cloudflare]"
	@echo "$(BOLD)Installing web frontend dependencies...$(RESET)"
	cd web && npm install
	@echo "$(GREEN)✓ Installation complete. Run 'source .venv/bin/activate' to activate.$(RESET)"

# ═══════════════════════════════════════════════════════════════
# Run TUI Dashboard
# ═══════════════════════════════════════════════════════════════

run: run-dev

run-dev:
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  APEX Risk Monitor - Development Mode$(RESET)"
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(YELLOW)  TUI Controls:$(RESET)"
	@echo "    $(GREEN)1-6$(RESET)  Switch views (Summary/Positions/Signals/Introspect/Data/Lab)"
	@echo "    $(GREEN)q$(RESET)    Quit"
	@echo "    $(GREEN)^C$(RESET)   Graceful shutdown"
	@echo ""
	@echo "$(YELLOW)  Connecting to IB Gateway (port 4001)...$(RESET)"
	@echo ""
	$(PYTHON) main.py --env dev --verbose

run-prod:
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  APEX Risk Monitor - Production Mode$(RESET)"
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	$(PYTHON) main.py --env prod

run-demo:
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  APEX Risk Monitor - Demo Mode (Offline)$(RESET)"
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(YELLOW)  Running with sample data (no broker connection required)$(RESET)"
	@echo ""
	$(PYTHON) main.py --env demo --verbose

run-headless:
	@echo "$(BOLD)Starting APEX in headless mode (no TUI)...$(RESET)"
	$(PYTHON) main.py --env dev --no-dashboard --verbose

# ═══════════════════════════════════════════════════════════════
# Code Quality
# ═══════════════════════════════════════════════════════════════

lint:
	@echo "$(BOLD)Checking code formatting...$(RESET)"
	$(VENV)/black --check src/ tests/
	$(VENV)/isort --check src/ tests/
	$(VENV)/flake8 src/ tests/

format:
	@echo "$(BOLD)Auto-formatting code...$(RESET)"
	$(VENV)/black src/ tests/
	$(VENV)/isort src/ tests/
	$(VENV)/flake8 src/ tests/
	@echo "$(GREEN)✓ Formatting complete$(RESET)"

type-check:
	@echo "$(BOLD)Running mypy type checker...$(RESET)"
	$(VENV)/mypy src/ tests/

# Hidden from help but still functional
dead-code:
	@echo "$(BOLD)Checking for dead code with vulture...$(RESET)"
	@$(VENV)/vulture src/ .vulture_whitelist.py --min-confidence 80 --exclude "src/legacy/,src/verification/" --sort-by-size || true
	@echo "$(YELLOW)Note: Review above for potential dead code to remove$(RESET)"

# Hidden from help but still functional
complexity:
	@echo "$(BOLD)Measuring code complexity...$(RESET)"
	@echo ""
	@echo "$(YELLOW)Cyclomatic Complexity (A=simple, F=very complex):$(RESET)"
	$(VENV)/radon cc src/ -a -nc --total-average
	@echo ""
	@echo "$(YELLOW)Maintainability Index (A=best, C=worst):$(RESET)"
	$(VENV)/radon mi src/ -s

quality: lint type-check dead-code complexity
	@echo "$(GREEN)✓ All quality checks complete$(RESET)"

# ═══════════════════════════════════════════════════════════════
# Testing
# ═══════════════════════════════════════════════════════════════

test:
	@echo "$(BOLD)Running unit tests...$(RESET)"
	$(VENV)/pytest tests/unit/ -v

test-all:
	@echo "$(BOLD)Running all tests...$(RESET)"
	$(VENV)/pytest

coverage:
	@echo "$(BOLD)Running tests with coverage...$(RESET)"
	$(VENV)/pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(RESET)"

# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

# PR gate - fast validation (10 symbols, 2 folds, no optuna)
validate-fast:
	@echo "$(BOLD)Running fast validation (PR gate)...$(RESET)"
	$(PYTHON) -m src.runners.validation_runner fast \
		--symbols SPY QQQ AAPL MSFT NVDA AMD MU GME AMC VIX \
		--timeframes 1d --folds 2 --no-optuna \
		--output reports/validation/fast

# ═══════════════════════════════════════════════════════════════
# Strategy Comparison
# ═══════════════════════════════════════════════════════════════

# Standalone strategy comparison (all strategies x universe symbols)
strategy-compare:
	@echo "$(BOLD)Running strategy comparison backtests...$(RESET)"
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--universe config/universe.yaml \
		--years 3 \
		--json-output out/signals/data/strategies.json
	@echo "$(GREEN)✓ JSON: out/signals/data/strategies.json$(RESET)"

# Quick strategy comparison (3 symbols, 3yr — needs 260+ bars for warmup)
strategy-compare-quick:
	@echo "$(BOLD)Quick strategy comparison (3 symbols, 3yr)...$(RESET)"
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--symbols SPY AAPL NVDA \
		--years 3 \
		--json-output /tmp/strategies_quick.json
	@echo "$(GREEN)✓ JSON: /tmp/strategies_quick.json$(RESET)"

# Full strategy verification — run after adding/modifying any strategy
# SOP: tests → type check → quick compare → inspect output
strategy-verify:
	@echo "$(BOLD)═══ Strategy Verification Pipeline ═══$(RESET)"
	@echo ""
	@echo "$(BOLD)Step 1/5: Unit tests (strategy objective + infrastructure)$(RESET)"
	$(PYTHON) -m pytest tests/unit/strategy/ -v --no-cov -q
	@echo ""
	@echo "$(BOLD)Step 2/5: Integration tests (signal parity)$(RESET)"
	$(PYTHON) -m pytest tests/integration/test_strategy_parity.py -v --no-cov -q
	@echo ""
	@echo "$(BOLD)Step 3/5: Type checking (strategy files)$(RESET)"
	$(PYTHON) -m mypy src/domain/strategy/signals/ src/backtest/optimization/strategy_objective.py --ignore-missing-imports
	@echo ""
	@echo "$(BOLD)Step 4/5: Formatting check$(RESET)"
	$(PYTHON) -m black --check src/domain/strategy/signals/ src/runners/strategy_compare_runner.py src/backtest/optimization/strategy_objective.py
	$(PYTHON) -m isort --check src/domain/strategy/signals/ src/runners/strategy_compare_runner.py src/backtest/optimization/strategy_objective.py
	@echo ""
	@echo "$(BOLD)Step 5/5: Quick compare (3 symbols, 3yr)$(RESET)"
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--symbols SPY AAPL NVDA \
		--years 3 \
		--json-output /tmp/strategies_verify.json
	@echo ""
	@echo "$(GREEN)═══ All strategy verification checks passed ═══$(RESET)"
	@echo "$(GREEN)✓ JSON: /tmp/strategies_verify.json$(RESET)"

# ═══════════════════════════════════════════════════════════════
# PEAD Earnings Drift Screener
# ═══════════════════════════════════════════════════════════════

# Full pipeline: earnings → attention → screen → track → resolve → stats
pead:
	@echo "$(BOLD)PEAD Earnings Drift Screen — Full Pipeline$(RESET)"
	$(PYTHON) -m src.runners.pead_runner --full \
		--universe config/universe.yaml
	@echo "$(GREEN)✓ Candidates: out/pead/data/pead_candidates.json$(RESET)"
	@echo "$(GREEN)✓ Tracker:    data/cache/pead_tracker.json$(RESET)"

# Screen from cached earnings (no FMP fetch — still tracks + resolves by default)
pead-screen:
	@echo "$(BOLD)PEAD Screen (from cache)...$(RESET)"
	$(PYTHON) -m src.runners.pead_runner --screen

# Run PEAD unit tests
pead-test:
	@echo "$(BOLD)Running PEAD screener tests...$(RESET)"
	$(PYTHON) -m pytest tests/unit/screeners/ -v --no-cov -q

.PHONY: pead pead-test pead-screen

# ═══════════════════════════════════════════════════════════════
# Momentum Screener
# ═══════════════════════════════════════════════════════════════

# Always-fresh: refresh universe + incremental OHLCV + screen
momentum:
	@echo "$(BOLD)Momentum Screen (refresh + screen)...$(RESET)"
	$(PYTHON) -m src.runners.momentum_runner

# Alias: momentum-update is now identical to momentum (refresh is default)
momentum-update: momentum

# Walk-forward backtest + ablation + HTML report
momentum-backtest:
	@echo "$(BOLD)Momentum Backtest + Ablation...$(RESET)"
	$(PYTHON) -m src.runners.momentum_runner --backtest --no-refresh
	@echo "$(GREEN)✓ Report: out/momentum/backtest.html$(RESET)"

# Run momentum screener tests
momentum-test:
	@echo "$(BOLD)Running momentum screener tests...$(RESET)"
	$(PYTHON) -m pytest tests/unit/screeners/test_momentum*.py -v --no-cov -q

.PHONY: momentum momentum-update momentum-backtest momentum-test

# Aliases (preferred names)
quantitative-moment: momentum
quantitative-moment-update: momentum-update
quantitative-moment-backtest: momentum-backtest
quantitative-moment-test: momentum-test

.PHONY: quantitative-moment quantitative-moment-update quantitative-moment-backtest quantitative-moment-test

# ═══════════════════════════════════════════════════════════════
# R2 Data Pipeline
# ═══════════════════════════════════════════════════════════════

r2-universe:   ## Build universe (FMP screener → R2 meta/)
	@echo "$(BOLD)Building universe → R2...$(RESET)"
	$(PYTHON) scripts/r2_universe_builder.py
	@echo "$(GREEN)✓ Universe uploaded to R2$(RESET)"

r2-backfill:   ## Full backfill (all symbols, 2019-present → R2 Parquet)
	@echo "$(BOLD)R2 historical backfill (2019-present, all symbols)...$(RESET)"
	$(PYTHON) scripts/r2_historical_loader.py --backfill
	@echo "$(GREEN)✓ Backfill complete$(RESET)"

r2-backfill-test:   ## Quick backfill test (5 symbols)
	@echo "$(BOLD)R2 backfill test (5 symbols)...$(RESET)"
	$(PYTHON) scripts/r2_historical_loader.py --backfill --symbols AAPL MSFT SPY QQQ NVDA
	@echo "$(GREEN)✓ Test backfill complete$(RESET)"

r2-delta:   ## Incremental delta update (last-bar + overlap → R2)
	@echo "$(BOLD)R2 delta update...$(RESET)"
	$(PYTHON) scripts/r2_historical_loader.py --delta
	@echo "$(GREEN)✓ Delta update complete$(RESET)"

r2-validate:   ## Generate data_quality.json only (no fetch)
	@echo "$(BOLD)R2 data quality validation...$(RESET)"
	$(PYTHON) scripts/r2_historical_loader.py --validate-only
	@echo "$(GREEN)✓ data_quality.json generated$(RESET)"

r2-market-caps:   ## Update market caps → R2 meta/market_caps.json
	@echo "$(BOLD)Updating market caps → R2...$(RESET)"
	$(PYTHON) scripts/r2_market_caps.py
	@echo "$(GREEN)✓ Market caps uploaded to R2$(RESET)"

.PHONY: r2-universe r2-backfill r2-backfill-test r2-delta r2-validate r2-market-caps

# ═══════════════════════════════════════════════════════════════
# Compute Jobs (same runners as /api/jobs/ endpoints)
# ═══════════════════════════════════════════════════════════════

jobs-momentum:   ## Run momentum screener
	@echo "$(BOLD)Running momentum screener...$(RESET)"
	$(PYTHON) -m src.runners.momentum_runner

jobs-pead:   ## Run PEAD screener
	@echo "$(BOLD)Running PEAD screener...$(RESET)"
	$(PYTHON) -m src.runners.pead_runner --full

jobs-strategy-compare:   ## Run strategy comparison backtest
	@echo "$(BOLD)Running strategy comparison...$(RESET)"
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--symbols SPY QQQ AAPL \
		--years 3

# ═══════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════

clean:
	@echo "$(BOLD)Cleaning build artifacts...$(RESET)"
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ .coverage coverage.xml
	rm -rf .ruff_cache/ .hypothesis/
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(RESET)"

# ── Live Dashboard ──────────────────────────────────────────

live:   ## Start full live dashboard (backend :8080 + frontend :5174)
	@echo "$(BOLD)Starting APEX Live Dashboard...$(RESET)"
	@echo "  Backend:  http://localhost:8080"
	@echo "  Frontend: http://localhost:5174"
	@echo "  Press Ctrl+C to stop both"
	@echo ""
	@# Kill stale processes on ports 8080/5174 if any
	@lsof -ti:8080 | xargs kill -9 2>/dev/null || true
	@lsof -ti:5174 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@# Ensure web dependencies are installed
	@if [ ! -d web/node_modules ]; then \
		echo "Installing web dependencies..." && \
		cd web && npm install; \
	fi
	@trap 'kill 0' EXIT; \
		PYTHONPATH=. $(PYTHON) -m uvicorn src.server.main:app --port 8080 & \
		echo "Waiting for backend to be ready (up to 45s)..." && \
		READY=0 && \
		for i in $$(seq 1 45); do \
			if curl -sf http://localhost:8080/api/health > /dev/null 2>&1; then \
				echo "Backend ready! ($$i s)"; \
				READY=1; \
				break; \
			fi; \
			sleep 1; \
		done && \
		if [ "$$READY" = "0" ]; then \
			echo "$(YELLOW)ERROR: Backend failed to start within 45s$(RESET)"; \
			exit 1; \
		fi && \
		cd web && npm run dev -- --port 5174

server-dev:   ## Start backend only (dev mode, auto-reload, :8080)
	@echo "$(BOLD)Starting APEX server (dev mode, auto-reload)...$(RESET)"
	PYTHONPATH=. $(PYTHON) -m uvicorn src.server.main:app --reload --port 8080

server:   ## Start backend only (production, :8080)
	@echo "$(BOLD)Starting APEX server (production)...$(RESET)"
	PYTHONPATH=. $(PYTHON) -m uvicorn src.server.main:app --port 8080

web-install:   ## Install web frontend dependencies
	@echo "$(BOLD)Installing web frontend dependencies...$(RESET)"
	cd web && npm install
	@echo "$(GREEN)✓ Web dependencies installed$(RESET)"

web-dev:   ## Start frontend only (dev mode, :5174)
	@echo "$(BOLD)Starting frontend dev server...$(RESET)"
	@if [ ! -d web/node_modules ]; then echo "Run 'make web-install' first" && exit 1; fi
	cd web && npm run dev

web-build:   ## Build frontend for production
	@echo "$(BOLD)Building frontend for production...$(RESET)"
	@if [ ! -d web/node_modules ]; then echo "Run 'make web-install' first" && exit 1; fi
	cd web && npm run build

tunnel:   ## Start Cloudflare Tunnel
	@echo "$(BOLD)Starting Cloudflare Tunnel...$(RESET)"
	cloudflared tunnel run apex
