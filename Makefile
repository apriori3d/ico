.PHONY: ci lint format typecheck pylance test

ci: lint typecheck pylance test

lint:
	poetry run ruff check src/
	poetry run ruff format --check src/

format:
	poetry run ruff format src/

typecheck:
	poetry run mypy src/

pylance:
	poetry run pyright src/

test:
	poetry run pytest --verbose
