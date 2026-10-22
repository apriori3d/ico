.PHONY: ci lint format typecheck pylance test

# The checks cover src/examples too: mypy needs the "examples" extra (skrub,
# ...), pyright also the "vision" one (torch for the CIFAR examples).
RUN = uv run --extra examples --extra vision

ci: lint typecheck pylance test

lint:
	$(RUN) ruff check src/
	$(RUN) ruff format --check src/

format:
	$(RUN) ruff format src/

typecheck:
	$(RUN) mypy src/

pylance:
	$(RUN) pyright src/

test:
	$(RUN) pytest --verbose
