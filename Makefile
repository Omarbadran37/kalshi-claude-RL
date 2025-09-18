# NFL Trading System Makefile

.PHONY: help install install-dev test test-unit test-integration lint format type-check clean docker-build docker-run

help: ## Show help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e .

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/ -m "not slow"

test-integration: ## Run integration tests only
	pytest tests/integration/

test-coverage: ## Run tests with coverage report
	pytest --cov=src/nfl_trading --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 src/ tests/
	black --check src/ tests/

format: ## Format code
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking
	mypy src/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker build -t nfl-trading-system .

docker-run: ## Run Docker container
	docker run -it --rm nfl-trading-system

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

setup-env: ## Set up development environment
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install -e .

run-dev: ## Run in development mode
	python -m src.nfl_trading.main --config configs/dev.yaml

run-prod: ## Run in production mode
	python -m src.nfl_trading.main --config configs/prod.yaml