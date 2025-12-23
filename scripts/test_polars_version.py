#!/usr/bin/env python3
"""Test polars_sim with basic functionality."""

import polars as pl
import polars_sim as ps


def main():
    """Run basic functionality test."""
    df1 = pl.DataFrame({"s": ["hello", "world"]})
    df2 = pl.DataFrame({"s": ["hello", "test"]})
    result = ps.join_sim(df1, df2, on="s", top_n=1, add_similarity=True)
    print(result)

    print(f"✓ Test passed with polars {pl.__version__}")
    print(f"  Result shape: {result.shape}")
    print(f"  Found {len(result)} matches")


if __name__ == "__main__":
    main()
