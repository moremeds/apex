# ═══════════════════════════════════════════════════════════════
# M2 Validation (Regime Detector)
# ═══════════════════════════════════════════════════════════════
#
# Usage:
#   make validate-test         # Quick test (~5 min)
#   make validate-full-publish # Full workflow + publish (~30-60 min)
#   make validate-smart        # Smart workflow (updates params only if gates pass)
#
# Individual steps:
#   make validate-optimize     # Optuna parameter optimization
#   make validate-full         # Full nested CV validation
#   make validate-holdout      # Holdout validation (release gate)
#   make validate-publish      # Publish to GitHub Pages
#
# ═══════════════════════════════════════════════════════════════

# Configuration
VALIDATION_UNIVERSE := config/validation/regime_universe.yaml
OPTIMIZED_PARAMS := config/validation/optimized_params.yaml
REPORTS_DIR := reports/validation

# ───────────────────────────────────────────────────────────────
# Individual Validation Steps
# ───────────────────────────────────────────────────────────────

validate-fast:
	@echo "$(BOLD)M2 Fast Validation (PR Gate)$(RESET)"
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON) -m src.runners.validation_runner fast \
		--symbols SPY QQQ AAPL MSFT NVDA AMD MU GME AMC TSLA \
		--timeframes 1d \
		--horizon-days 20 \
		--days 500 \
		--output $(REPORTS_DIR)/fast_validation.json
	@echo "$(GREEN)✓ Fast validation complete$(RESET)"

validate-full:
	@echo "$(BOLD)M2 Full Validation (Nightly Gate)$(RESET)"
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON) -m src.runners.validation_runner full \
		--universe $(VALIDATION_UNIVERSE) \
		--timeframes 1d \
		--horizon-days 20 \
		--outer-folds 5 \
		--inner-folds 3 \
		--inner-trials 20 \
		--days 750 \
		--max-symbols 50 \
		--output $(REPORTS_DIR)/full_validation.json
	@echo "$(GREEN)✓ Full validation complete$(RESET)"

validate-optimize:
	@echo "$(BOLD)M2 Parameter Optimization (Nested CV + Optuna)$(RESET)"
	@mkdir -p $(REPORTS_DIR) config/validation
	$(PYTHON) -m src.runners.validation_runner optimize \
		--universe $(VALIDATION_UNIVERSE) \
		--outer-folds 3 \
		--inner-folds 2 \
		--inner-trials 30 \
		--days 750 \
		--max-symbols 30 \
		--horizon-days 20 \
		--output $(REPORTS_DIR)/optimization_result.json \
		--params-output $(OPTIMIZED_PARAMS)
	@echo "$(GREEN)✓ Optimization complete: $(OPTIMIZED_PARAMS)$(RESET)"

validate-holdout:
	@echo "$(BOLD)M2 Holdout Validation (Release Gate)$(RESET)"
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON) -m src.runners.validation_runner holdout \
		--universe $(VALIDATION_UNIVERSE) \
		--horizon-days 20 \
		--days 500 \
		--output $(REPORTS_DIR)/holdout_validation.json
	@echo "$(GREEN)✓ Holdout validation complete$(RESET)"

validate-all: validate-optimize validate-full validate-holdout
	@echo "$(GREEN)✓ All validation steps complete$(RESET)"

# ───────────────────────────────────────────────────────────────
# Quick Test (Minimal Symbols)
# ───────────────────────────────────────────────────────────────

validate-test:
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)M2 MINIMAL VALIDATION TEST$(RESET)"
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@mkdir -p $(REPORTS_DIR) config/validation
	@echo ""
	@echo "$(YELLOW)Step 1/5: Parameter Optimization (minimal)$(RESET)"
	$(PYTHON) -m src.runners.validation_runner optimize \
		--universe $(VALIDATION_UNIVERSE) \
		--outer-folds 2 --inner-folds 2 --inner-trials 5 \
		--days 400 --max-symbols 8 --horizon-days 20 \
		--output $(REPORTS_DIR)/test_optimization.json \
		--params-output $(REPORTS_DIR)/test_optimized_params.yaml || true
	@echo ""
	@echo "$(YELLOW)Step 2/5: Full Validation (minimal)$(RESET)"
	$(PYTHON) -m src.runners.validation_runner full \
		--universe $(VALIDATION_UNIVERSE) \
		--timeframes 1d --horizon-days 20 \
		--outer-folds 2 --inner-folds 2 --inner-trials 5 \
		--days 400 --max-symbols 10 \
		--output $(REPORTS_DIR)/test_full_validation.json || true
	@echo ""
	@echo "$(YELLOW)Step 3/5: Holdout Validation$(RESET)"
	$(PYTHON) -m src.runners.validation_runner holdout \
		--universe $(VALIDATION_UNIVERSE) \
		--horizon-days 20 --days 400 \
		--output $(REPORTS_DIR)/test_holdout_validation.json || true
	@echo ""
	@echo "$(YELLOW)Step 4/5: Signal Report (package format)$(RESET)"
	$(PYTHON) -m src.runners.signal_runner --live \
		--symbols SPY QQQ AAPL MSFT NVDA \
		--timeframes 1d 4h 1h \
		--format package \
		--html-output $(REPORTS_DIR)/signal_report || true
	@echo ""
	@echo "$(YELLOW)Step 5/5: Validation Summary Report$(RESET)"
	$(PYTHON) -m src.domain.signals.reporting.validation_report \
		--reports-dir $(REPORTS_DIR) \
		--output $(REPORTS_DIR)/validation_summary.html || true
	@cp $(REPORTS_DIR)/validation_summary.html $(REPORTS_DIR)/signal_report/validation.html 2>/dev/null || true
	@echo ""
	@echo "$(GREEN)✓ Test complete. View: cd $(REPORTS_DIR)/signal_report && python -m http.server 8080$(RESET)"

# ───────────────────────────────────────────────────────────────
# Publish to GitHub Pages
# ───────────────────────────────────────────────────────────────

validate-publish:
	@echo "$(BOLD)Publishing Reports to GitHub Pages$(RESET)"
	@mkdir -p $(REPORTS_DIR)/publish
	@# Copy signal package (index.html inside)
	@if [ -d "$(REPORTS_DIR)/signal_report" ]; then \
		cp -r $(REPORTS_DIR)/signal_report/* $(REPORTS_DIR)/publish/; \
	elif [ -f "$(REPORTS_DIR)/signal_report.html" ]; then \
		cp $(REPORTS_DIR)/signal_report.html $(REPORTS_DIR)/publish/index.html; \
	fi
	@# Copy validation summary
	@cp $(REPORTS_DIR)/validation_summary.html $(REPORTS_DIR)/publish/validation.html 2>/dev/null || true
	@cp $(REPORTS_DIR)/*.json $(REPORTS_DIR)/publish/ 2>/dev/null || true
	@touch $(REPORTS_DIR)/publish/.nojekyll
	@# Deploy
	@cd $(REPORTS_DIR)/publish && \
		git init -q && git checkout -q -b gh-pages && git add . && \
		git commit -q -m "Deploy reports - $$(date '+%Y-%m-%d %H:%M')" && \
		git push -f origin gh-pages 2>/dev/null || \
		echo "$(YELLOW)Run manually: cd $(REPORTS_DIR)/publish && git push -f <remote> gh-pages$(RESET)"
	@echo "$(GREEN)✓ Published: index.html -> validation.html$(RESET)"
	@ls -la $(REPORTS_DIR)/publish/

# ───────────────────────────────────────────────────────────────
# Full Workflow + Publish
# ───────────────────────────────────────────────────────────────

validate-full-publish:
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)M2 FULL VALIDATION + PUBLISH$(RESET)"
	@echo "$(BOLD)═══════════════════════════════════════════════════════════════$(RESET)"
	@mkdir -p $(REPORTS_DIR) config/validation
	@echo ""
	@echo "$(YELLOW)Step 1/6: Parameter Optimization$(RESET)"
	$(PYTHON) -m src.runners.validation_runner optimize \
		--universe $(VALIDATION_UNIVERSE) \
		--outer-folds 3 --inner-folds 2 --inner-trials 20 \
		--days 600 --max-symbols 25 --horizon-days 20 \
		--output $(REPORTS_DIR)/optimization_result.json \
		--params-output $(OPTIMIZED_PARAMS) || true
	@echo ""
	@echo "$(YELLOW)Step 2/6: Full Validation$(RESET)"
	$(PYTHON) -m src.runners.validation_runner full \
		--universe $(VALIDATION_UNIVERSE) \
		--timeframes 1d --horizon-days 20 \
		--outer-folds 3 --inner-folds 2 --inner-trials 10 \
		--days 600 --max-symbols 40 \
		--output $(REPORTS_DIR)/full_validation.json || true
	@echo ""
	@echo "$(YELLOW)Step 3/6: Holdout Validation$(RESET)"
	$(PYTHON) -m src.runners.validation_runner holdout \
		--universe $(VALIDATION_UNIVERSE) \
		--horizon-days 20 --days 500 \
		--output $(REPORTS_DIR)/holdout_validation.json || true
	@echo ""
	@echo "$(YELLOW)Step 4/6: Validation Summary Report$(RESET)"
	$(PYTHON) -m src.domain.signals.reporting.validation_report \
		--reports-dir $(REPORTS_DIR) \
		--output $(REPORTS_DIR)/validation_summary.html || true
	@echo ""
	@echo "$(YELLOW)Step 5/6: Signal Report (full universe)$(RESET)"
	$(PYTHON) -m src.runners.signal_runner --live \
		--universe $(VALIDATION_UNIVERSE) \
		--timeframes 1d 4h 1h \
		--format package \
		--html-output $(REPORTS_DIR)/signal_report || true
	@cp $(REPORTS_DIR)/validation_summary.html $(REPORTS_DIR)/signal_report/validation.html 2>/dev/null || true
	@echo ""
	@echo "$(YELLOW)Step 6/6: Publish to GitHub Pages$(RESET)"
	@$(MAKE) validate-publish
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)✓ FULL VALIDATION + PUBLISH COMPLETE$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(RESET)"

# ───────────────────────────────────────────────────────────────
# Smart Workflow (conditional param update)
# ───────────────────────────────────────────────────────────────

validate-smart:
	@echo "$(BOLD)M2 SMART VALIDATION WORKFLOW$(RESET)"
	@echo "Updates params ONLY if all gates pass"
	$(PYTHON) scripts/validation_workflow.py --mode full --reports-dir $(REPORTS_DIR)
	@$(MAKE) validate-publish

validate-smart-test:
	@echo "$(BOLD)M2 SMART VALIDATION (Test Mode)$(RESET)"
	$(PYTHON) scripts/validation_workflow.py --mode test --reports-dir $(REPORTS_DIR) --force
	@$(MAKE) validate-publish

.PHONY: validate-fast validate-full validate-optimize validate-holdout validate-all \
        validate-test validate-publish validate-full-publish validate-smart validate-smart-test
