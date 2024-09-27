from polars_sim._polars_sim import awesome_cossim
import polars as pl

__version__ = "0.1.0"

__all__ = ["join_sim"]


def join_sim(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    ntop: int = 1,
    normalize: bool = False,
    threads: int | None = None,
    suffix: str = "_right",
    add_mapping: bool = False,
    add_similarity: bool = True,
) -> pl.DataFrame:
    """
    Compute the cosine similarity between two DataFrames.

    Parameters:

        left (pl.DataFrame): The left DataFrame.
        right (pl.DataFrame): The right DataFrame.
        left_on (str): The column name in the left DataFrame to be used for similarity computation.
        right_on (str): The column name in the right DataFrame to be used for similarity computation.
        ntop (int): The number of top similarities to return.
        normalize (bool): Whether to l2-normalize the vectors before computing the cosine similarity. Defaults to False.
            If True, you will get the cosine similarity of the vectorizations. Otherweise, you will get the dot product
            and hence the number of common non-zero elements.
        threads (int | None, optional): The number of threads to use for computation. Defaults to None.
        add_mapping (bool): Whether to add the row-col mapping of df_left and df_right to the result. Defaults to False.
        add_similarity (bool): Whether to add the similarity score to the result. Defaults to True.

    Returns:
        pl.DataFrame: A DataFrame containing the cosine similarity results.
    """

    if on is not None:
        left_on = on
        right_on = on

    assert left_on in left.columns, f"{left_on} not in left DataFrame"
    assert right_on in right.columns, f"{right_on} not in right DataFrame"

    # TODO: Find the dtype that matches both rust and the polars version
    # call the rust function
    # indices are uint by default
    _map = awesome_cossim(
        left, right, left_on, right_on, ntop, threads, normalize
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
