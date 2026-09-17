# ETA Prediction System - Makefile
#
# Quick Start:
#   make build     # Build Docker image
#   make test      # Run tests
#   make up        # Start full stack
#   make shell     # Interactive shell in container

.PHONY: help build up down test shell logs clean prefect dev

# Default target
help:
	@echo "ETA Prediction System"
	@echo ""
	@echo "Usage:"
	@echo "  make build      Build Docker image"
	@echo "  make up         Start Redis + run tests"
	@echo "  make down       Stop all services"
	@echo "  make test       Run test suite"
	@echo "  make shell      Interactive shell in container"
	@echo "  make logs       View service logs"
	@echo "  make prefect    Start Prefect flow"
	@echo "  make dev        Development mode with hot reload"
	@echo "  make clean      Remove containers and volumes"
	@echo ""
	@echo "Redis only:"
	@echo "  make redis      Start Redis only"
	@echo "  make redis-cli  Connect to Redis CLI"

# Build the Docker image
build:
	docker compose build

# Start Redis and run default (tests)
up: build
	docker compose up

# Start Redis only
redis:
	docker compose up -d redis

# Connect to Redis CLI
redis-cli:
	docker compose exec redis redis-cli

# Stop all services
down:
	docker compose down

# Run tests
test: build
	docker compose run --rm eta pytest core/tests/test_core.py -v

# Run all tests
test-all: build
	docker compose run --rm eta pytest -v

# Interactive shell
shell: build
	docker compose run --rm eta bash

# View logs
logs:
	docker compose logs -f

# Start Prefect flow
prefect: build
	docker compose --profile prefect up

# Development mode
dev: build
	docker compose --profile dev run --rm eta-dev

# Clean up everything
clean:
	docker compose down -v --rmi local
	docker system prune -f

# Quick verification that everything works
verify: build
	@echo "==> Starting Redis..."
	docker compose up -d redis
	@echo "==> Waiting for Redis..."
	sleep 2
	@echo "==> Running tests..."
	docker compose run --rm eta pytest core/tests/test_core.py -v
	@echo "==> Testing estimator import..."
	docker compose run --rm eta python -c "from eta_service.estimator import estimate_stop_times; print('Estimator OK')"
	@echo "==> Cleaning up..."
	docker compose down
	@echo ""
	@echo "All checks passed!"

# ---------------------------------------------------------------------------
# Local (non-Docker) test targets
#
# The suite CANNOT run in a single pytest process: torch and xgboost fight over
# native OpenMP on macOS, and exercising both in one interpreter segfaults
# inside xgboost's DMatrix label construction. It is not a test bug -- it
# reproduces with just models/tests/test_lstm.py plus any xgboost-touching test
# file, and with KMP_DUPLICATE_LIB_OK=TRUE set. So the deep-learning tests run
# in their own process.
#
# PY_DL is the dedicated Python 3.12 environment the LSTM needs: torch dropped
# macOS x86_64 wheels after 2.2.2, which in turn requires CPython <=3.12, while
# the repo's default .venv is 3.14. See models/lstm/README.md.
# ---------------------------------------------------------------------------
PY_DL := PYTHONPATH=.:models .venv-dl/bin/python

.PHONY: test-local test-models test-lstm test-features

# Everything except the torch tests (one process).
test-models:
	$(PY_DL) -m pytest models/tests --ignore=models/tests/test_lstm.py -q

# The torch tests, isolated (separate process -- see the note above).
test-lstm:
	$(PY_DL) -m pytest models/tests/test_lstm.py -q

# feature_engineering + core run under the DEFAULT venv, not PY_DL: they need
# Django, etaval and rt_pipeline, which .venv-dl deliberately does not carry.
# test_dataset_builder.py is excluded because it imports the sibling
# gtfs-django repo's `gtfs` app through sch_pipeline.models -- the standalone
# -reproducibility gap roadmap 1.8(a) exists to close, not a broken test.
test-features:
	PYTHONPATH=.:gtfs-rt-pipeline uv run --group bucr --extra train \
		--with pytest --with duckdb python -m pytest \
		feature_engineering/tests core/tests -q \
		--ignore=feature_engineering/tests/test_dataset_builder.py

# The full local suite, in the process split that actually works.
test-local: test-models test-lstm test-features
