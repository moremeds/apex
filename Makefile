# APEX Development Makefile
# Quick commands for common development tasks

.PHONY: install run run-dev run-prod run-demo run-headless lint format type-check dead-code complexity quality test test-all coverage clean help validate-fast validate signals-test dashboard-test dashboard-data dashboard-data-ready dashboard-signal dashboard-signal-qa dashboard-web-dev dashboard-web-preview-deploy signals signals-deploy strategy-compare strategy-verify strategy-compare-quick behavioral behavioral-full behavioral-cases pead pead-test pead-screen r2-universe r2-backfill r2-backfill-test r2-delta r2-validate server-dev server web-install web-dev web-build live tunnel

# Virtual environment - use .venv/bin executables directly
VENV := .venv/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip

# Colors for terminal output
BOLD := $(shell tput bold)
RESET := $(shell tput sgr0)
GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)

# Dashboard preview branch for Cloudflare Pages (non-production)
DASHBOARD_PREVIEW_BRANCH ?= dashboard-preview

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
	@echo "  make validate       Full validation suite"
	@echo ""
	@echo "$(GREEN)Signal Pipeline:$(RESET)"
	@echo "  make signals-test      Quick test (20 symbols) + strategy comparison + HTTP server"
	@echo "  make signals           Full production run (all features)"
	@echo "  make signals-deploy    Deploy to GitHub Pages"
	@echo "  make strategy-compare  Run strategy backtests + comparison dashboard"
	@echo "  make strategy-compare-quick  Quick compare (3 symbols, 1yr)"
	@echo "  make strategy-verify   Full strategy verification (tests + compare + lint)"
	@echo ""
	@echo "$(GREEN)Behavioral Gate:$(RESET)"
	@echo "  make behavioral       Quick test (default params) + serve"
	@echo ""
	@echo "$(GREEN)TrendPulse:$(RESET)"
	@echo "  make tp-validate      Full 3-stage validation (36 symbols)"
	@echo "  make tp-holdout       Holdout only (faster)"
	@echo "  make tp-optimize      Phase 1 Optuna optimization"
	@echo "  make tp-universe      Full universe backtest + HTML report"
	@echo "  make tp-universe-quick  Quick test (12 symbols)"
	@echo "  make behavioral-full  Optuna optimization + walk-forward + serve"
	@echo "  make behavioral-cases Predefined case studies + serve"
	@echo ""
	@echo "$(GREEN)PEAD Screener:$(RESET)"
	@echo "  make pead           Full pipeline (caps + earnings + attention + screen + track + stats)"
	@echo "  make pead-test      Run PEAD unit tests"
	@echo "  make pead-screen    Screen from cache (still tracks + resolves by default)"
	@echo ""
	@echo "$(GREEN)Momentum Screener:$(RESET)"
	@echo "  make quantitative-moment          Refresh + screen + email (alias: momentum)"
	@echo "  make quantitative-moment-update   Same as above (refresh is now default)"
	@echo "  make quantitative-moment-backtest Walk-forward backtest + ablation + HTML"
	@echo "  make quantitative-moment-test     Run unit tests"
	@echo ""
	@echo "$(GREEN)Dashboard Data Pipeline:$(RESET)"
	@echo "  make dashboard-data-ready  Validate & backfill OHLCV data coverage"
	@echo "  make dashboard-signal      Generate signal report (full universe)"
	@echo "  make dashboard-signal-qa   Run quality gates (G1-G15)"
	@echo "  make dashboard-data        Full pipeline (signals + screeners + backtests)"
	@echo ""
	@echo "$(GREEN)Dashboard Web Build & Deploy:$(RESET)"
	@echo "  make dashboard-build   Build CF dashboard from pipeline output"
	@echo "  make dashboard-dev     Build + serve locally (:8801)"
	@echo "  make dashboard-web-dev Build + serve with Wrangler Pages dev (:8801)"
	@echo "  make dashboard-web-preview-deploy Build + deploy preview branch ($(DASHBOARD_PREVIEW_BRANCH))"
	@echo "  make dashboard-deploy  Build + deploy to Cloudflare Pages"
	@echo ""
	@echo "$(GREEN)R2 Data Pipeline:$(RESET)"
	@echo "  make r2-universe       Screen universe (FMP → filter → ~500 symbols → R2)"
	@echo "  make r2-backfill       Full backfill (all symbols, 2019-present)"
	@echo "  make r2-backfill-test  Quick test (5 symbols)"
	@echo "  make r2-delta          Incremental delta update"
	@echo "  make r2-validate       Generate data_quality.json only"
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

# Full validation suite (signal report + quality gates + turning points)
# Uses unified config/universe.yaml
validate:
	@echo "$(BOLD)Running full validation suite...$(RESET)"
	@echo "Step 1/3: Generating signal report..."
	$(PYTHON) -m src.runners.signal_runner --live \
		--universe config/universe.yaml \
		--subset pr_validation \
		--timeframes 1d 4h 1h \
		--preload-concurrency 5 \
		--parallel-writes 8 \
		--format package \
		--html-output reports/validation/signal_report
	@echo "Step 2/3: Running quality gates (G1-G10)..."
	$(PYTHON) scripts/validate_gates.py --all \
		--package reports/validation/signal_report -v
	@echo "Step 3/3: Validating turning points..."
	$(PYTHON) scripts/validate_turning_points.py \
		--config tests/fixtures/turning_point_samples.yaml --all
	@echo "$(GREEN)✓ All validation checks complete$(RESET)"

# ═══════════════════════════════════════════════════════════════
# Signal Pipeline
# ═══════════════════════════════════════════════════════════════

# Standalone strategy comparison (all strategies x universe symbols)
strategy-compare:
	@echo "$(BOLD)Running strategy comparison backtests...$(RESET)"
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--universe config/universe.yaml \
		--years 3 \
		--output out/signals/strategies.html \
		--json-output out/signals/data/strategies.json
	@echo "$(GREEN)✓ Dashboard: out/signals/strategies.html$(RESET)"

# Quick strategy comparison (3 symbols, 3yr — needs 260+ bars for warmup)
strategy-compare-quick:
	@echo "$(BOLD)Quick strategy comparison (3 symbols, 3yr)...$(RESET)"
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--symbols SPY AAPL NVDA \
		--years 3 \
		--output /tmp/strategies_quick.html
	@echo "$(GREEN)✓ Dashboard: /tmp/strategies_quick.html$(RESET)"

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
		--output /tmp/strategies_verify.html
	@echo ""
	@echo "$(GREEN)═══ All strategy verification checks passed ═══$(RESET)"
	@echo "$(GREEN)✓ Dashboard: /tmp/strategies_verify.html$(RESET)"

# Quick test with 12 diverse symbols + HTTP server
signals-test:
	@echo "$(BOLD)Quick signal test (20 symbols)...$(RESET)"
	$(PYTHON) -m src.runners.signal_runner --live \
		--symbols SPY QQQ XLB GLD TLT UVXY AAPL NVDA JPM XOM UNH HD DIS TSLA AMD META SLV SNDK MU ORCL\
		--timeframes 1d 1h 4h\
		--preload-concurrency 5 --parallel-writes 8 \
		--format package \
		--html-output /tmp/signal_test
	@echo ""
	@echo "$(BOLD)Running strategy comparison backtests...$(RESET)"
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--symbols SPY QQQ AAPL NVDA JPM XOM UNH HD DIS TSLA \
		--years 3 \
		--output /tmp/signal_test/strategies.html
	@echo ""
	@echo "$(GREEN)Starting HTTP server at http://localhost:8800$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	cd /tmp/signal_test && python3 -m http.server 8800

# DEPRECATED: Use 'make dashboard-data && make dashboard-dev' instead.
dashboard-test: dashboard-data dashboard-build
	@lsof -ti:8801 | xargs kill -9 2>/dev/null || true
	@echo "$(BOLD)Serving dashboard at http://localhost:8801$(RESET)"
	@open http://localhost:8801 &
	python3 -m http.server 8801 --directory out/site

dashboard-data-ready:  ## Validate & backfill OHLCV data coverage
	@echo "$(BOLD)Validating data coverage...$(RESET)"
	$(PYTHON) scripts/historical_data_loader.py validate --timeframes 1d,4h,1h
	@echo ""
	@echo "$(BOLD)Backfilling gaps (if any)...$(RESET)"
	$(PYTHON) scripts/historical_data_loader.py backfill --timeframes 1d,4h,1h --source fmp
	@echo ""
	@echo "$(BOLD)Generating coverage report...$(RESET)"
	$(PYTHON) scripts/historical_data_loader.py report --output out/coverage/coverage_report.html
	@echo "$(GREEN)✓ Data ready$(RESET)"

dashboard-signal:  ## Generate signal report (full universe, no deploy)
	@echo "$(BOLD)Generating signals (full universe)...$(RESET)"
	$(PYTHON) -m src.runners.signal_runner --live \
		--universe config/universe.yaml \
		--timeframes 1d 1h 4h \
		--preload-concurrency 5 \
		--parallel-writes 8 \
		--format package \
		--html-output out/signals
	@echo "$(GREEN)✓ Signals ready in out/signals/$(RESET)"

dashboard-signal-qa:  ## Run signal quality gates (G1-G15)
	@echo "$(BOLD)Running quality gates...$(RESET)"
	$(PYTHON) scripts/validate_gates.py --all --package out/signals
	@echo "$(GREEN)✓ QA gates passed$(RESET)"

dashboard-data: dashboard-signal  ## Full data pipeline (signals + screeners + strategy compare)
	@echo ""
	@echo "Step 2/3: Screeners (optional)..."
	$(PYTHON) -m src.runners.momentum_runner --screen --no-refresh || true
	$(PYTHON) -m src.runners.pead_runner --screen || true
	@echo ""
	@echo "Step 3/3: Strategy comparison..."
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--universe config/universe.yaml \
		--years 3 \
		--output out/signals/strategies.html \
		--json-output out/signals/data/strategies.json
	@echo "$(GREEN)✓ Dashboard data ready$(RESET)"

# Dashboard data-only pipeline: generate package inputs without serving/deploying (20 symbols)
dashboard-data-quick:
	@echo "$(BOLD)Dashboard data pipeline (20 symbols, no serve/deploy)...$(RESET)"
	@echo ""
	@echo "Step 1/4: Signal generation..."
	$(PYTHON) -m src.runners.signal_runner --live \
		--symbols SPY QQQ XLB GLD TLT UVXY AAPL NVDA JPM XOM UNH HD DIS TSLA AMD META SLV IWM DIA MU \
		--timeframes 1d 1h 4h \
		--preload-concurrency 5 \
		--parallel-writes 8 \
		--format package \
		--html-output out/signals
	@echo ""
	@echo "Step 2/4: Momentum screener (optional, from cache if available)..."
	$(PYTHON) -m src.runners.momentum_runner --screen --no-refresh || true
	@echo ""
	@echo "Step 3/4: PEAD screener (optional, from cache if available)..."
	$(PYTHON) -m src.runners.pead_runner --screen || true
	@echo ""
	@echo "Step 4/4: Strategy comparison backtests..."
	$(PYTHON) -m src.runners.strategy_compare_runner \
		--symbols SPY QQQ AAPL NVDA JPM XOM UNH HD DIS TSLA \
		--years 3 \
		--output out/signals/strategies.html \
		--json-output out/signals/data/strategies.json
	@echo ""
	@echo "$(GREEN)✓ Dashboard data ready in out/signals/$(RESET)"

# Full production run with ALL features (update-caps, retrain, heatmap, validate)
# Uses unified config/universe.yaml for all operations
signals:
	@echo "$(BOLD)Full signal pipeline...$(RESET)"
	@echo "Step 1/3: Updating market caps..."
	$(PYTHON) -m src.runners.signal_runner --update-market-caps \
		--universe config/universe.yaml
	@echo "Step 2/3: Retraining models (full universe)..."
	$(PYTHON) -m src.runners.signal_runner --retrain-models \
		--universe config/universe.yaml
	@echo "Step 3/3: Generating report..."
	$(PYTHON) -m src.runners.signal_runner --live \
		--universe config/universe.yaml \
		--timeframes 1d 4h 1h \
		--preload-concurrency 5 --parallel-writes 8 \
		--format package \
		--html-output out/signals
	@echo "$(GREEN)✓ Report: out/signals/index.html$(RESET)"
	@echo "$(GREEN)✓ Heatmap: out/signals/heatmap.html$(RESET)"
	@echo "$(BOLD)Serving at http://localhost:8800 — Ctrl+C to stop$(RESET)"
	@cd out/signals && python3 -m http.server 8800

# Deploy to GitHub Pages (full pipeline + deploy in one command)
signals-deploy:
	@echo "$(BOLD)Generating and deploying signal report to GitHub Pages...$(RESET)"
	@echo "Step 1/3: Updating market caps..."
	$(PYTHON) -m src.runners.signal_runner --update-market-caps \
		--universe config/universe.yaml
	@echo "Step 2/3: Retraining models (full universe)..."
	$(PYTHON) -m src.runners.signal_runner --retrain-models \
		--universe config/universe.yaml
	@echo "Step 3/3: Generating report and deploying..."
	$(PYTHON) -m src.runners.signal_runner --live \
		--universe config/universe.yaml \
		--timeframes 1d 4h 1h \
		--preload-concurrency 5 --parallel-writes 8 \
		--format package \
		--html-output out/signals \
		--deploy github
	@echo "$(GREEN)✓ Report deployed to GitHub Pages$(RESET)"

# Deploy existing out/signals directory (no regeneration)
signals-push:
	@echo "$(BOLD)Deploying existing out/signals to GitHub Pages...$(RESET)"
	@test -d out/signals || (echo "$(RED)Error: out/signals not found. Run 'make signals' first.$(RESET)" && exit 1)
	$(eval REMOTE_URL := $(shell git remote get-url origin))
	@rm -rf /tmp/gh-pages-deploy
	@cp -r out/signals /tmp/gh-pages-deploy
	@cd /tmp/gh-pages-deploy && \
		git init -b gh-pages && \
		git add -A && \
		git commit -m "Deploy signal report $$(date +%Y-%m-%d_%H-%M)" && \
		git remote add origin $(REMOTE_URL) && \
		git push -f origin gh-pages
	@rm -rf /tmp/gh-pages-deploy
	@echo "$(GREEN)✓ Deployed to GitHub Pages$(RESET)"


# Quick deploy (skip retraining - use existing models)
signals-deploy-quick:
	@echo "$(BOLD)Quick deploy (skip retraining)...$(RESET)"
	@echo "Step 1/2: Updating market caps..."
	$(PYTHON) -m src.runners.signal_runner --update-market-caps \
		--universe config/universe.yaml
	@echo "Step 2/2: Generating report and deploying..."
	$(PYTHON) -m src.runners.signal_runner --live \
		--universe config/universe.yaml \
		--timeframes 1d 4h 1h \
		--preload-concurrency 5 --parallel-writes 8 \
		--format package \
		--html-output out/signals \
		--deploy github
	@echo "$(GREEN)✓ Report deployed to GitHub Pages$(RESET)"

# ═══════════════════════════════════════════════════════════════
# Behavioral Gate Validation
# ═══════════════════════════════════════════════════════════════

# Quick test: default params, universe quick_test, 2018-2025, then serve
behavioral:
	@echo "$(BOLD)Behavioral gate test (quick_test subset, 2018-2025)...$(RESET)"
	$(PYTHON) -m src.backtest.runner --behavioral \
		--start 2018-01-01 --end 2025-12-31
	@echo "$(GREEN)✓ Report: out/behavioral/$(RESET)"
	@echo "$(BOLD)Serving at http://localhost:8081$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@cd out/behavioral && python3 -m http.server 8081

# Predefined case studies (market episodes)
behavioral-cases:
	@echo "$(BOLD)Running predefined behavioral case studies...$(RESET)"
	$(PYTHON) -m src.backtest.runner --behavioral-cases
	@echo "$(GREEN)✓ Reports: out/behavioral/$(RESET)"
	@echo "$(BOLD)Serving at http://localhost:8081$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@cd out/behavioral && python3 -m http.server 8081

# Full pipeline: Optuna optimization + walk-forward + auto-clustering
behavioral-full:
	@echo "$(BOLD)Full behavioral gate optimization pipeline...$(RESET)"
	$(PYTHON) -m src.backtest.runner --behavioral --cluster \
		--spec config/backtest/dual_macd_behavioral.yaml
	@echo "$(GREEN)✓ Reports: out/behavioral/$(RESET)"
	@echo "$(GREEN)✓ Cluster candidate: config/gate_policy_clusters.yaml$(RESET)"
	@echo "$(BOLD)Serving at http://localhost:8081$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@cd out/behavioral && python3 -m http.server 8081

# ═══════════════════════════════════════════════════════════════
# TrendPulse v2.2 Validation
# ═══════════════════════════════════════════════════════════════

tp-validate:
	@echo "$(BOLD)TrendPulse v2.2 — Full 3-stage validation (36 symbols)$(RESET)"
	$(PYTHON) scripts/trend_pulse_validate.py

tp-holdout:
	@echo "$(BOLD)TrendPulse v2.2 — Holdout only$(RESET)"
	$(PYTHON) scripts/trend_pulse_validate.py --skip-full

tp-optimize:
	@echo "$(BOLD)TrendPulse v2.2 — Phase 1 Optuna optimization$(RESET)"
	$(PYTHON) -m src.backtest.runner \
		--spec config/backtest/examples/trend_pulse_phase1.yaml

tp-universe:
	@echo "$(BOLD)TrendPulse v2.2 — Full universe backtest + HTML report$(RESET)"
	$(PYTHON) scripts/trend_pulse_universe.py
	@echo "$(GREEN)✓ Report: out/trend_pulse/universe_report.html$(RESET)"

tp-universe-quick:
	@echo "$(BOLD)TrendPulse v2.2 — Quick test (12 symbols)$(RESET)"
	$(PYTHON) scripts/trend_pulse_universe.py --subset quick_test \
		--output out/trend_pulse/quick_report.html
	@echo "$(GREEN)✓ Report: out/trend_pulse/quick_report.html$(RESET)"

.PHONY: tp-validate tp-holdout tp-optimize tp-universe tp-universe-quick

# ═══════════════════════════════════════════════════════════════
# PEAD Earnings Drift Screener
# ═══════════════════════════════════════════════════════════════

# Full pipeline: caps → earnings → attention → screen → track → resolve → stats → HTML
pead:
	@echo "$(BOLD)PEAD Earnings Drift Screen — Full Pipeline$(RESET)"
	@echo ""
	@echo "$(BOLD)Step 1/2: Updating market caps...$(RESET)"
	$(PYTHON) -m src.runners.signal_runner --update-market-caps \
		--universe config/universe.yaml
	@echo ""
	@echo "$(BOLD)Step 2/2: Full PEAD run (earnings + attention + screen + track + stats)...$(RESET)"
	$(PYTHON) -m src.runners.pead_runner --full \
		--universe config/universe.yaml \
		--html-output out/pead/pead.html
	@echo ""
	@echo "$(GREEN)✓ Candidates: out/pead/data/pead_candidates.json$(RESET)"
	@echo "$(GREEN)✓ Tracker:    data/cache/pead_tracker.json$(RESET)"
	@echo "$(GREEN)✓ HTML report: out/pead/pead.html$(RESET)"

# Screen from cached earnings (no FMP fetch — still tracks + resolves by default)
pead-screen:
	@echo "$(BOLD)PEAD Screen (from cache)...$(RESET)"
	$(PYTHON) -m src.runners.pead_runner --screen \
		--html-output out/pead/pead.html
	@echo "$(GREEN)✓ HTML report: out/pead/pead.html$(RESET)"

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
	$(PYTHON) -m src.runners.momentum_runner --html out/momentum/report.html
	@echo "$(GREEN)✓ HTML report: out/momentum/report.html$(RESET)"

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
# Cloudflare Dashboard
# ═══════════════════════════════════════════════════════════════

dashboard-build:   ## Build CF dashboard from existing pipeline output
	@echo "$(BOLD)Building Cloudflare dashboard...$(RESET)"
	$(PYTHON) -c "from src.infrastructure.reporting.dashboard import DashboardBuilder; DashboardBuilder().build()"
	@echo "$(GREEN)✓ Dashboard built → out/site/$(RESET)"

dashboard-web-dev: dashboard-build   ## Build + serve with Wrangler Pages dev (:8801)
	@echo "$(BOLD)Serving dashboard (Wrangler Pages dev) at http://localhost:8801$(RESET)"
	npx wrangler@3 pages dev out/site --port 8801

dashboard-web-preview-deploy: dashboard-build   ## Build + deploy to Cloudflare preview branch
	@echo "$(BOLD)Deploying dashboard preview branch '$(DASHBOARD_PREVIEW_BRANCH)'...$(RESET)"
	npx wrangler@3 pages deploy out/site/ --project-name apex-dashboard --branch $(DASHBOARD_PREVIEW_BRANCH)

dashboard-dev: dashboard-build   ## Build + serve locally (:8801)
	@echo "$(BOLD)Serving dashboard at http://localhost:8801$(RESET)"
	python3 -m http.server 8801 --directory out/site

dashboard-deploy: dashboard-build   ## Build + deploy to Cloudflare Pages
	npx wrangler@3 pages deploy out/site/ --project-name apex-dashboard

.PHONY: dashboard-build dashboard-dev dashboard-deploy dashboard-data dashboard-web-dev dashboard-web-preview-deploy live

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

.PHONY: r2-universe r2-backfill r2-backfill-test r2-delta r2-validate

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
