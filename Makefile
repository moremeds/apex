# APEX Development Makefile
# Quick commands for common development tasks

.PHONY: install lint format type-check dead-code complexity quality test test-all coverage clean help diagrams diagrams-classes diagrams-deps diagrams-flows validate-fast validate signals-test signals signals-deploy

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
	@echo "  make signals-test   Quick test (12 symbols) + HTTP server"
	@echo "  make signals        Full production run (all features)"
	@echo "  make signals-deploy Deploy to GitHub Pages"
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
		--timeframes 1d --folds 2 --no-optuna

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

# Quick test with 12 diverse symbols + HTTP server
signals-test:
	@echo "$(BOLD)Quick signal test (12 symbols)...$(RESET)"
	$(PYTHON) -m src.runners.signal_runner --live \
		--symbols SPY QQQ AAPL NVDA JPM XOM UNH HD DIS TSLA AMD META \
		--timeframes 1d \
		--format package \
		--html-output /tmp/signal_test
	@echo ""
	@echo "$(GREEN)Starting HTTP server at http://localhost:8080$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	cd /tmp/signal_test && python3 -m http.server 8080

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

# Deploy to GitHub Pages (full pipeline + deploy in one command)
signals-deploy:
	@echo "$(BOLD)Generating and deploying signal report to GitHub Pages...$(RESET)"
	@echo "Step 1/3: Updating market caps..."
	$(PYTHON) -m src.runners.signal_runner --update-market-caps \
		--universe config/universe.yaml
	@echo "Step 2/3: Retraining models (full universe)..."
	$(PYTHON) -m src.runners.signal_runner --retrain-models \
		--universe config/universe.yaml
	@echo "Step 3/3: Generating report and deploying to GitHub Pages..."
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

# Serve local report (for testing before deploy)
signals-serve:
	@echo "$(BOLD)Serving signal report at http://localhost:8080$(RESET)"
	@test -d out/signals || (echo "$(RED)Error: out/signals not found. Run 'make signals' first.$(RESET)" && exit 1)
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@cd out/signals && python3 -m http.server 8080

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
