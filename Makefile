# APEX Development Makefile
# Quick commands for common development tasks

.PHONY: install run run-dev run-prod run-demo run-headless lint format type-check dead-code complexity quality test test-all coverage clean help diagrams diagrams-classes diagrams-deps diagrams-flows validate-fast validate signals-test signals signals-deploy strategy-compare strategy-verify strategy-compare-quick behavioral behavioral-full behavioral-cases pead pead-test pead-screen

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
	@echo "$(GREEN)Other:$(RESET)"
	@echo "  make diagrams       Generate architecture diagrams"
	@echo "  make clean          Remove build artifacts"

# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

install:
	@echo "$(BOLD)Installing dependencies with uv...$(RESET)"
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev,observability]"
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
		--output out/signals/strategies.html
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

# Run momentum screen from cached data
momentum:
	@echo "$(BOLD)Momentum Screen (from cache)...$(RESET)"
	$(PYTHON) -m src.runners.momentum_runner --html out/momentum/report.html
	@echo "$(GREEN)✓ HTML report: out/momentum/report.html$(RESET)"

# Fetch fresh data + screen
momentum-update:
	@echo "$(BOLD)Momentum Screen — Full Update...$(RESET)"
	$(PYTHON) -m src.runners.momentum_runner --update --html out/momentum/report.html
	@echo "$(GREEN)✓ HTML report: out/momentum/report.html$(RESET)"

# Walk-forward backtest
momentum-backtest:
	@echo "$(BOLD)Momentum Backtest...$(RESET)"
	$(PYTHON) -m src.runners.momentum_runner --backtest --html out/momentum/backtest.html
	@echo "$(GREEN)✓ Backtest: out/momentum/backtest.html$(RESET)"

# Run momentum screener tests
momentum-test:
	@echo "$(BOLD)Running momentum screener tests...$(RESET)"
	$(PYTHON) -m pytest tests/unit/screeners/test_momentum*.py -v --no-cov -q

.PHONY: momentum momentum-update momentum-backtest momentum-test

# ═══════════════════════════════════════════════════════════════
# Diagrams
# ═══════════════════════════════════════════════════════════════

diagrams: diagrams-classes diagrams-deps diagrams-flows
	@echo "$(GREEN)✓ All diagrams generated in docs/diagrams/$(RESET)"

# Hidden from help but still functional
diagrams-classes:
	@echo "$(BOLD)Generating class diagrams (PlantUML via pyreverse)...$(RESET)"
	@mkdir -p docs/diagrams/classes
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p domain_services src/domain/services
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p domain_signals src/domain/signals
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p domain_events src/domain/events
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p infrastructure_adapters src/infrastructure/adapters
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p application_orchestrator src/application/orchestrator
	@echo "$(GREEN)✓ Class diagrams generated$(RESET)"

# Hidden from help but still functional
diagrams-deps:
	@echo "$(BOLD)Generating dependency graphs (SVG)...$(RESET)"
	@mkdir -p docs/diagrams/dependencies
	$(VENV)/pydeps src/domain -o docs/diagrams/dependencies/domain_deps.svg --max-module-depth=2 --cluster --noshow
	$(VENV)/pydeps src/infrastructure -o docs/diagrams/dependencies/infrastructure_deps.svg --max-module-depth=2 --cluster --noshow
	$(VENV)/pydeps src -o docs/diagrams/dependencies/full_project_deps.svg --max-module-depth=1 --cluster --noshow
	@echo "$(GREEN)✓ Dependency graphs generated$(RESET)"

# Hidden from help but still functional
diagrams-flows:
	@echo "$(BOLD)Generating call flow diagrams (SVG via code2flow)...$(RESET)"
	@mkdir -p docs/diagrams/flows
	$(VENV)/code2flow src/application/orchestrator -o docs/diagrams/flows/orchestrator_flow.svg --language py --no-trimming --quiet
	$(VENV)/code2flow src/domain/services/risk -o docs/diagrams/flows/risk_services_flow.svg --language py --no-trimming --quiet
	$(VENV)/code2flow src/domain/signals -o docs/diagrams/flows/signal_pipeline_flow.svg --language py --no-trimming --quiet
	@echo "$(GREEN)✓ Call flow diagrams generated$(RESET)"

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
