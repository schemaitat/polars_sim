# polars_sim

## Description

Implements an **approximate join** of two polars dataframes based on string columns.


Right now, we use a fixed vectorization, which is applied on the fly and eventually
used in a sparse matrix multiplication combined with a top-n selection. This produces
the cosine similarities of the individual string pairs.

The `join_sim` function is similar to `join_asof` but for strings instead of timestamps.

## Installation

```bash
pip install polars_sim
```

## Development

We use [uv](https://docs.astral.sh/uv/) for python package management. Furthermore, you need rust to be installed, see [install rust](https://www.rust-lang.org/tools/install). You won't need to activate an enviroment by yourself at any point. This is handled by uv. To get started, run
```bash
# create a virtual environment
uv venv --seed -p 3.11
# install dependencies
uv pip install -e .
# install dev dependencies
uv pip install -r requirements.txt
# compiple rust code
make install 
# run tests
make test
```

## Usage

```python
import polars as pl
import polars_sim as ps

df_left = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "David"],
    }
)

df_right = pl.DataFrame(
    {
        "name": ["Ali", "Alice in Wonderland", "Bobby", "Tom"],
    }
)

df = ps.join_sim(
    df_left,
    df_right,
    on="name",
    ntop=4,
)

shape: (3, 3)
┌───────┬──────────┬─────────────────────┐
│ name  ┆ sim      ┆ name_right          │
│ ---   ┆ ---      ┆ ---                 │
│ str   ┆ f64      ┆ str                 │
╞═══════╪══════════╪═════════════════════╡
│ Alice ┆ 0.57735  ┆ Ali                 │
│ Alice ┆ 0.522233 ┆ Alice in Wonderland │
│ Bob   ┆ 0.57735  ┆ Bobby               │
└───────┴──────────┴─────────────────────┘
```

# References

The implementation is based on an algorithm used in [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn), which itself is an improvement of the scipy sparse matrix multiplication.
