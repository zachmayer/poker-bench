check-requirements:
	uv --version
.PHONY: check-requirements

install: check-requirements
	uv python install
	uv sync --locked --all-groups
	uv run pre-commit install
.PHONY: install

upgrade:
	uv lock --upgrade
	uv run pre-commit autoupdate
.PHONY: upgrade

lint:
	uv run pre-commit run -a
.PHONY: lint

types:
	uv run pyright .
.PHONY: types

test:
	uv run pytest
.PHONY: test

clean:
	rm uv.lock
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .ipynb_checkpoints
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf .coverage.xml
.PHONY: clean

all: lint types test
