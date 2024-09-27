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

## Usage

```python
import polars as pl
import polars_sim as ps

df_left = pl.DataFrame({
    "name" : ["Alice", "Bob", "Charlie", "David"],
})

df_right = pl.DataFrame({
    "name" : ["Ali", "Bobby", "Tom"],
})

df = ps.join_sim(
    df_left,
    df_right,
    on="name",
    ntop=2,
)

shape: (2, 5)
┌─────┬───────┬─────┬──────┬────────────┐
│ row ┆ name  ┆ col ┆ data ┆ name_right │
│ --- ┆ ---   ┆ --- ┆ ---  ┆ ---        │
│ u32 ┆ str   ┆ u32 ┆ i64  ┆ str        │
╞═════╪═══════╪═════╪══════╪════════════╡
│ 0   ┆ Alice ┆ 1   ┆ 2    ┆ Alicia     │
│ 1   ┆ Bob   ┆ 2   ┆ 1    ┆ Bobby      │
└─────┴───────┴─────┴──────┴────────────┘
```

# Notes

The implementation is based on an algorithm used in [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn), which itself is an improvement of the scipy sparse matrix multiplication.
