# VideoAnnotator v1.2.0 - Modern Development Workflow
# Uses uv for fast, reliable package management

PY := uv run
UV := export PATH="$$HOME/.local/bin:$$PATH" && uv

# Default target
.PHONY: help
help:
	@echo "VideoAnnotator v1.2.0 - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install dependencies with uv"
	@echo "  install-dev    Install dev dependencies and setup pre-commit"
	@echo "  install-torch-cuda  Install PyTorch with CUDA support"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with ruff"
	@echo "  typecheck      Run mypy type checking"
	@echo "  quality-check  Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run pytest test suite"
	@echo "  test-fast      Run fast unit tests (~30s)"
	@echo "  test-integration  Run integration tests (~5min)"
	@echo "  test-all       Run complete test suite"
	@echo "  test-cov       Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  server         Start API server"
	@echo "  server-dev     Start API server with auto-reload"
	@echo "  demo           Run basic demo"
	@echo "  demo-batch     Run batch processing demo"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build CPU Docker image"
	@echo "  docker-build-gpu Build GPU Docker image"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean build artifacts and cache"
	@echo "  check-env      Check environment status"
	@echo "  lock           Update lockfile"
	@echo "  sync           Sync environment with lockfile"
	@echo "  help           Show this help message"

# Installation and environment
.PHONY: install
install:
	$(UV) sync

.PHONY: install-dev
install-dev: install
	$(UV) add --dev ruff mypy pytest pre-commit
	$(PY) pre-commit install

# Testing commands
.PHONY: test
test:
	$(PY) pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

.PHONY: test-unit
test-unit:
	$(PY) pytest tests/ -v -m "unit" --cov=src --cov-report=term-missing

.PHONY: test-integration
test-integration:
	$(PY) pytest tests/ -v -m "integration" --cov=src --cov-report=term-missing

.PHONY: test-performance
test-performance:
	$(PY) pytest tests/ -v -m "performance" --benchmark-only

.PHONY: test-fast
test-fast:
	$(PY) python scripts/test_fast.py

.PHONY: test-all
test-all:
	$(PY) python scripts/test_all.py

# Code quality commands (now using Ruff)
.PHONY: lint
lint:
	$(PY) ruff check .

.PHONY: format
format:
	$(PY) ruff format .

.PHONY: format-check
format-check:
	$(PY) ruff format --check .

.PHONY: type-check
type-check:
	$(PY) mypy src

.PHONY: quality-check
quality-check: lint format-check type-check
	@echo "All quality checks passed!"

# Build commands
.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.PHONY: build
build: clean
	$(UV) build

# API server commands
.PHONY: server
server:
	$(PY) python api_server.py

.PHONY: server-dev
server-dev:
	$(PY) uvicorn api_server:app --host 0.0.0.0 --port 18011 --reload

# PyTorch CUDA installation helper
.PHONY: install-torch-cuda
install-torch-cuda:
	@echo "Installing PyTorch with CUDA support..."
	@echo "Check https://pytorch.org/get-started/locally/ for latest versions"
	$(UV) add "torch>=2.4.0" "torchvision>=0.19.0" "torchaudio>=2.4.0" --index-url https://download.pytorch.org/whl/cu124

# Environment helpers
.PHONY: check-env
check-env:
	@echo "UV Version: $$($(UV) --version || echo 'uv not found')"
	@echo "Python Version: $$($(PY) python --version || echo 'Python not available')"

.PHONY: lock
lock:
	$(UV) lock

.PHONY: sync
sync:
	$(UV) sync

# Docker commands
docker-build:
	docker build -t videoannotator:latest .

docker-build-gpu:
	docker build --target gpu-base -t videoannotator:gpu .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/output:/app/output videoannotator:latest

docker-run-gpu:
	docker run --gpus all --rm -v $(PWD)/data:/app/data -v $(PWD)/output:/app/output videoannotator:gpu

docker-dev:
	docker-compose --profile dev up --build

docker-jupyter:
	docker-compose --profile dev up jupyter

# Documentation commands
docs:
	@echo "Building documentation..."
	@mkdir -p docs/build
	@echo "Documentation build completed!"

serve-docs:
	@echo "Serving documentation on http://localhost:18011"
	@cd docs && python -m http.server 18011

# Analysis commands
benchmark:
	pytest tests/ -v -m "performance" --benchmark-only --benchmark-json=benchmark.json
	@echo "Benchmark results saved to benchmark.json"

security-scan:
	bandit -r src/
	safety check
	pip-audit

# Pre-commit setup
setup-pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

pre-commit:
	pre-commit run --all-files

# Example usage commands
example-basic:
	python examples/basic_video_processing.py

example-batch:
	python examples/batch_processing.py

example-pipelines:
	python examples/test_individual_pipelines.py

example-config:
	python examples/custom_pipeline_config.py

# CLI examples
demo-default:
	uv run python -m src.cli --help

demo-server:
	uv run python -m src.cli server --help

demo-version:
	uv run python -m src.cli version

# Development environment setup
dev-setup: install-dev setup-pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything is working."

# Production environment setup
prod-setup: install
	@echo "Production environment setup complete!"

# Full development cycle
dev-cycle: quality-check test
	@echo "Development cycle complete - ready for commit!"

# Release preparation
release-prep: clean quality-check test build
	@echo "Release preparation complete!"
	@echo "Review the build artifacts in dist/ before publishing."

# Quick start for new developers
quickstart:
	@echo "VideoAnnotator Quick Start"
	@echo "=========================="
	@echo ""
	@echo "1. Setting up development environment..."
	$(MAKE) dev-setup
	@echo ""
	@echo "2. Running tests to verify installation..."
	$(MAKE) test-unit
	@echo ""
	@echo "3. Quick start complete! Try these commands:"
	@echo "   make demo-default          # Show CLI help"
	@echo "   make example-basic         # Run basic example"
	@echo "   make test                  # Run full test suite"
	@echo "   make quality-check         # Check code quality"
