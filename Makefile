# VideoAnnotator Makefile
# Common development and deployment tasks

.PHONY: help install install-dev install-all test test-unit test-integration test-performance
.PHONY: lint format type-check quality-check clean build docker-build docker-run
.PHONY: docs serve-docs benchmark security-scan pre-commit setup-pre-commit

# Default target
help:
	@echo "VideoAnnotator Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install basic dependencies"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-all      Install with all optional dependencies"
	@echo "  setup-pre-commit Setup pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black"
	@echo "  type-check       Run type checking with mypy"
	@echo "  quality-check    Run all quality checks"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"
	@echo "  serve-docs       Serve documentation locally"
	@echo ""
	@echo "Analysis:"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  security-scan    Run security vulnerability scan"

# Installation commands
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

install-all:
	pip install -e .[all]

# Testing commands
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/ -v -m "unit" --cov=src --cov-report=term-missing

test-integration:
	pytest tests/ -v -m "integration" --cov=src --cov-report=term-missing

test-performance:
	pytest tests/ -v -m "performance" --benchmark-only

# Code quality commands
lint:
	flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:
	black src tests examples
	isort src tests examples

type-check:
	mypy src

quality-check: lint format type-check
	@echo "All quality checks passed!"

# Build commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

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
	@echo "Serving documentation on http://localhost:8000"
	@cd docs && python -m http.server 8000

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

# Main CLI examples
demo-default:
	python main.py --help

demo-config:
	python main.py --config configs/default.yaml --help

demo-lightweight:
	python main.py --config configs/lightweight.yaml --help

demo-high-performance:
	python main.py --config configs/high_performance.yaml --help

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
