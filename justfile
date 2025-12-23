set shell := ["bash", "-c"]

# Install Python dependencies
install-py:
    uv sync

# Install development version
install: install-py
    uv run maturin develop --uv

# Clean up build artifacts and environment
destroy:
    rm -rf .venv
    rm -rf target
    rm -rf dist

# Install release version
install-release:
    uv run maturin develop --release

# Run pre-commit hooks
pre-commit:
    uv run pre-commit run --all-files

# Run tests
test:
    uv run pytest tests

# Test plugin with a specific polars version
test-polars-version version:
    uv run --no-project --isolated \
        --with polars=={{version}} \
        --with pyarrow \
        --with dist/polars_sim-*.whl \
        python scripts/test_polars_version.py

# Test plugin against multiple polars versions in parallel
test-polars-matrix n="all":
    bash scripts/run_matrix_tests.sh {{n}}

# Edit benchmark notebook
edit-bench:
    uv run --with-requirements benchmark/requirements.txt \
        marimo edit benchmark/bench.py

# Run benchmark and export to HTML
run-bench:
    uv run --with-requirements benchmark/requirements.txt \
        marimo export html \
        --output benchmark/bench.html \
        benchmark/bench.py -- \
        -size_left 5000 \
        -size_right 100000