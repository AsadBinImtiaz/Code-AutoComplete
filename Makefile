# Code Autocomplete Assistant - Development Makefile

.PHONY: help setup install test lint format clean server

# Default target
help:
	@echo "Available targets:"
	@echo "  setup     - Create virtual environment and install dependencies"
	@echo "  install   - Install package in development mode"
	@echo "  test      - Run all tests"
	@echo "  lint      - Run linting (flake8, mypy)"
	@echo "  format    - Format code (black)"
	@echo "  server    - Start development server"
	@echo "  clean     - Clean up generated files"

# Environment setup
setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"
	@echo "Then run: make install"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	pytest -v --tb=short

# Run tests with coverage
test-coverage:
	pytest --cov=server --cov=config --cov=scripts --cov-report=html --cov-report=term

# Linting
lint:
	flake8 server config scripts evaluation
	mypy server config scripts evaluation

# Code formatting
format:
	black server config scripts evaluation tests
	isort server config scripts evaluation tests

# Start development server
server:
	uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Create project config template
config-template:
	python -c "from config.settings import config_manager; print('Created:', config_manager.create_project_config_template('.'))"

# Initialize git repository
git-init:
	git init
	git add .
	git commit -m "Initial project setup"