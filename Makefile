SHELL=/bin/bash

install:
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release:
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit:
	cargo fmt --all && cargo clippy --all-features
	uv tool run ruff check . --fix --exit-non-zero-on-fix
	uv tool run ruff format python tests

test:
	source .venv/bin/activate  && pytest tests

run: install
	source .venv/bin/activate && python run.py

run-release: install-release
	source .venv/bin/activate && python run.py

edit-bench:
	uv run --with-requirements benchmark/requirements.txt marimo edit benchmark/bench.py 