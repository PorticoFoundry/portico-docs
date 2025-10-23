.PHONY: help serve build clean validate test check install

help:
	@echo "Portico Documentation - Available Commands"
	@echo ""
	@echo "  make serve       - Start local dev server with live reload (http://127.0.0.1:8000)"
	@echo "  make build       - Build static site to site/"
	@echo "  make clean       - Remove built site directory"
	@echo "  make validate    - Validate documentation coverage"
	@echo "  make test        - Run documentation tests"
	@echo "  make check       - Run all checks (validate + test + build)"
	@echo "  make install     - Install dependencies with poetry"
	@echo ""

serve:
	@echo "Starting MkDocs dev server..."
	poetry run mkdocs serve

build:
	@echo "Building static site..."
	poetry run mkdocs build

build-strict:
	@echo "Building static site (strict mode - warnings as errors)..."
	poetry run mkdocs build --strict

clean:
	@echo "Cleaning built site..."
	rm -rf site/

validate:
	@echo "Validating documentation coverage..."
	poetry run python scripts/validate_docs_coverage.py

test:
	@echo "Running documentation tests..."
	poetry run pytest docs_tests/ -v

check: validate test build-strict
	@echo ""
	@echo "✅ All documentation checks passed!"

install:
	@echo "Installing dependencies..."
	poetry install
