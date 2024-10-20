from typing import Literal
import polars as pl
from polars_sim._polars_sim import awesome_cossim


def join_sim(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    top_n: int = 1,
    normalization: Literal["l2", "count"] = "l2",
    threads: int | None = None,
    suffix: str = "_right",
    add_mapping: bool = False,
    add_similarity: bool = True,
    threading_dimension: Literal["left", "right", "auto"] = "auto",
) -> pl.DataFrame:
    """
    Compute the cosine similarity between two DataFrames.

    Parameters:

        left (pl.DataFrame): The left DataFrame.
        right (pl.DataFrame): The right DataFrame.
        left_on (str): The column name in the left DataFrame to be used for similarity computation.
        right_on (str): The column name in the right DataFrame to be used for similarity computation.
        ntop (int): The number of top similarities to return.
        normalization (Literal["l2", "count]): If "l2", the vectors are l2-normalized and the similarity is the cosine
            similarity. If "count", the similarity is the number of common elements/tokens.
        threads (int | None, optional): The number of threads to use for computation. Defaults to None.
        add_mapping (bool): Whether to add the row-col mapping of df_left and df_right to the result. Defaults to False.
        add_similarity (bool): Whether to add the similarity score to the result. Defaults to True.
        threading_dimension (Literal["left", "right", "auto"]): The dimension to parallelize. If "left", the left DataFrame
            is parallelized. If "right", the right DataFrame is parallelized.

    Returns:
        pl.DataFrame: A DataFrame containing the cosine similarity results.
    """

    if on is not None:
        left_on = on
        right_on = on

    assert left_on in left.columns, f"{left_on} not in left DataFrame"
    assert right_on in right.columns, f"{right_on} not in right DataFrame"

    normalize: bool = True

    match threading_dimension:
        case "left":
            parallelize_left = True
        case "right":
            parallelize_left = False
        case "auto":
            if len(left) < len(right) * 10:
                parallelize_left = True
            else:
                parallelize_left = False
        case _:
            raise ValueError(f"Threading {threading_dimension} not supported")

    match normalization:
        case "l2":
            normalize = True
        case "count":
            normalize = False
        case _:
            raise ValueError(f"Normalization {normalization} not supported")

    # TODO: Find the dtype that matches both rust and the polars version
    # call the rust function
    _map = awesome_cossim(
        left, right, left_on, right_on, top_n, threads, normalize, parallelize_left
    ).cast(
        {
            "row": pl.UInt32,
            "col": pl.UInt32,
        }
    )

    return (
        left.with_row_index("row")
        .join(_map, on="row", how="left")
        .join(right.with_row_index("col"), on="col", suffix=suffix)
        .pipe(lambda df: df.drop("row", "col") if not add_mapping else df)
        .pipe(lambda df: df.drop("sim") if not add_similarity else df)
    )
