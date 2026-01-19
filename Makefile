# APEX Development Makefile
# Quick commands for common development tasks

.PHONY: install lint format type-check test test-all coverage clean dead-code complexity quality verify help diagrams diagrams-classes diagrams-deps diagrams-flows

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
	@echo "  make install       Install all dependencies with uv"
	@echo ""
	@echo "$(GREEN)Quality:$(RESET)"
	@echo "  make lint          Check formatting (black, isort, flake8)"
	@echo "  make format        Auto-fix formatting (black, isort)"
	@echo "  make type-check    Run mypy type checker"
	@echo "  make dead-code     Find unused code with vulture"
	@echo "  make complexity    Measure code complexity with radon"
	@echo "  make quality       Run all quality checks"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  make test          Run unit tests"
	@echo "  make test-all      Run all tests (unit + integration)"
	@echo "  make coverage      Run tests with HTML coverage report"
	@echo ""
	@echo "$(GREEN)Verification:$(RESET)"
	@echo "  make verify        Run signal and regime verification"
	@echo ""
	@echo "$(GREEN)Documentation:$(RESET)"
	@echo "  make diagrams      Generate all architecture diagrams"
	@echo "  make diagrams-classes  Generate class diagrams (PlantUML)"
	@echo "  make diagrams-deps     Generate dependency graphs (SVG)"
	@echo "  make diagrams-flows    Generate call flow diagrams (SVG)"
	@echo ""
	@echo "$(GREEN)Cleanup:$(RESET)"
	@echo "  make clean         Remove build artifacts and caches"

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
	$(VENV)/mypy src/

dead-code:
	@echo "$(BOLD)Checking for dead code with vulture...$(RESET)"
	@$(VENV)/vulture src/ .vulture_whitelist.py --min-confidence 80 --exclude "src/legacy/,src/verification/" --sort-by-size || true
	@echo "$(YELLOW)Note: Review above for potential dead code to remove$(RESET)"

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
# Verification
# ═══════════════════════════════════════════════════════════════

verify:
	@echo "$(BOLD)Running signal verification...$(RESET)"
	$(PYTHON) -m src.verification.signal_verifier --all --profile signal_dev
	@echo ""
	@echo "$(BOLD)Running regime verification...$(RESET)"
	$(PYTHON) -m src.verification.regime_verifier --all --profile dev

# ═══════════════════════════════════════════════════════════════
# Diagrams
# ═══════════════════════════════════════════════════════════════

diagrams: diagrams-classes diagrams-deps diagrams-flows
	@echo "$(GREEN)✓ All diagrams generated in docs/diagrams/$(RESET)"

diagrams-classes:
	@echo "$(BOLD)Generating class diagrams (PlantUML via pyreverse)...$(RESET)"
	@mkdir -p docs/diagrams/classes
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p domain_services src/domain/services
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p domain_signals src/domain/signals
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p domain_events src/domain/events
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p infrastructure_adapters src/infrastructure/adapters
	$(VENV)/pyreverse -o puml -d docs/diagrams/classes -p application_orchestrator src/application/orchestrator
	@echo "$(GREEN)✓ Class diagrams generated$(RESET)"

diagrams-deps:
	@echo "$(BOLD)Generating dependency graphs (SVG)...$(RESET)"
	@mkdir -p docs/diagrams/dependencies
	$(VENV)/pydeps src/domain -o docs/diagrams/dependencies/domain_deps.svg --max-module-depth=2 --cluster --noshow
	$(VENV)/pydeps src/infrastructure -o docs/diagrams/dependencies/infrastructure_deps.svg --max-module-depth=2 --cluster --noshow
	$(VENV)/pydeps src -o docs/diagrams/dependencies/full_project_deps.svg --max-module-depth=1 --cluster --noshow
	@echo "$(GREEN)✓ Dependency graphs generated$(RESET)"

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
