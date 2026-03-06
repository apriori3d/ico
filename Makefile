.PHONY: ci lint format typecheck test

ci: lint typecheck test

lint:
	poetry run ruff check src/
	poetry run ruff format --check src/

format:
	poetry run ruff format src/

typecheck:
	poetry run mypy src/

test:
	poetry run pytest --verbose
