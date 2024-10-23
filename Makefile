SHELL=/bin/bash

install-py:
	uv sync

install: install-py
	uv run maturin develop

destroy:
	rm -rf .venv
	rm -rf target
	rm -rf dist

install-release:
	uv run maturin develop --release

pre-commit:
	cargo fmt --all 
	cargo clippy --all-features
	uv run pre-commit run --all-files

test:
	uv run pytest tests

run: install
	uv run run.py

run-release: install-release
	uv run run.py

edit-bench:
	uv run --with-requirements benchmark/requirements.txt marimo edit benchmark/bench.py

run-bench:
	uv run --with-requirements benchmark/requirements.txt marimo export html --output benchmark/bench.html benchmark/bench.py -- -size_left 5000 -size_right 100000