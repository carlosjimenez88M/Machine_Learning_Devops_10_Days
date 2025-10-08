.PHONY: setup install test run clean help

PYTHON := python3.11
VENV := venv
VENV_BIN := $(VENV)/bin

help:
	@echo "Available commands:"
	@echo "  make setup      - Create virtual environment with Python 3.11"
	@echo "  make install    - Install all dependencies"
	@echo "  make test       - Run unit tests"
	@echo "  make run        - Start the FastAPI server"
	@echo "  make clean      - Remove virtual environment and cache files"

setup:
	@echo "Creating virtual environment with Python 3.11..."
	@if ! command -v python3.11 &> /dev/null; then \
		echo "Error: Python 3.11 not found. Please install Python 3.11 first."; \
		exit 1; \
	fi
	python3.11 -m venv $(VENV)
	@echo "✓ Virtual environment created"

install: setup
	@echo "Installing dependencies..."
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e ../loggers_configuration
	$(VENV_BIN)/pip install -r requirements.txt
	@echo "✓ Dependencies installed"

test:
	@echo "Running tests..."
	$(VENV_BIN)/pytest tests/ -v
	@echo "✓ Tests completed"

run:
	@echo "Starting FastAPI server on http://localhost:8000"
	@echo "API documentation: http://localhost:8000/docs"
	$(VENV_BIN)/python main.py

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup completed"
