# polars_simed

## Description

Implements an **approximate join** of two polars dataframes based on string columns.


Right now, we use a fixed vectorization, which is applied on the fly and eventually
used in a sparse matrix multiplication combined with a top-n selection. This produces
the cosine similarities of the individual string pairs.

The `join_sim` function is similar to a left join or `join_asof` but for strings instead of timestamps.

## Installation

```bash
pip install polars_simed
```

## Development

We use [uv](https://docs.astral.sh/uv/) for python package management. Furthermore, you need rust to be installed, see [install rust](https://www.rust-lang.org/tools/install). You won't need to activate an enviroment by yourself at any point. This is handled by uv. To get started, run
```bash
# install python dependencies and compile the rust code
make install 
# run tests
make test
```

## Usage

```python
import polars as pl
import polars_simed as ps

df_left = pl.DataFrame(
    {
        "name": ["alice", "bob", "charlie", "david"],
    }
)

df_right = pl.DataFrame(
    {
        "name": ["ali", "alice in wonderland", "bobby", "tom"],
    }
)

df = ps.join_sim(
    df_left,
    df_right,
    on="name",
    top_n=4,
)

shape: (3, 3)
┌───────┬──────────┬─────────────────────┐
│ name  ┆ sim      ┆ name_right          │
│ ---   ┆ ---      ┆ ---                 │
│ str   ┆ f32      ┆ str                 │
╞═══════╪══════════╪═════════════════════╡
│ alice ┆ 0.57735  ┆ ali                 │
│ alice ┆ 0.522233 ┆ alice in wonderland │
│ bob   ┆ 0.57735  ┆ bobby               │
└───────┴──────────┴─────────────────────┘
```

# Performance

A benchmark can be executed with `make run-bench`. 
In general, the performance heavily depends on the length of the dataframes.
By default, the computation is parallelized over the left dataframe. However, serveral benchmarks 
showed that if the right dataframe is much bigger than the left dataframe and no normalization is applied, it is faster to parallelize over the right dataframe. 

If no normalization is applied, the performance is usually better since the a small uint type will
be used for the sparse matrix multiplication, e.g. u16. Otherwise, all types will be of 32 bit size.

# References

The implementation is based on an algorithm used in [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn), which itself is an improvement of the scipy sparse matrix multiplication.
