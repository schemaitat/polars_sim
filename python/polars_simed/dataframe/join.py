from typing import Literal
import polars as pl
from polars_simed._polars_simed_lib import awesome_cossim


def normalize_string_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col)
        .str.replace_all("[^a-zA-Z0-9]", "")
        .str.to_lowercase()
        .alias(f"{col}_normalized"),
    )


def join_sim(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    top_n: int = 10,
    normalization: Literal["l2", "count"] = "l2",
    apply_word_normalization: bool = False,
    threads: int | None = None,
    suffix: str = "_right",
    add_mapping: bool = False,
    add_similarity: bool = True,
    threading_dimension: Literal["left", "right", "auto"] = "left",
) -> pl.DataFrame:
    """
    Compute the row-wise similarity between two DataFrames. Behaves similar to a
    left join or join_asof for time series data.

    The similarity is the inner product of the (normalized) three-gram representation
    of the strings.

    Parameters:

        left (pl.DataFrame):
            The left DataFrame.
        right (pl.DataFrame):
            The right DataFrame.

        left_on (str):
            The column name in the left DataFrame to be used for similarity computation.

        right_on (str):
            The column name in the right DataFrame to be used for similarity computation.

        top_n (int):
            The maximum number of similar items to return for each row.

        normalization (Literal["l2", "count]):
            If "l2", the vectors are l2-normalized and the similarity is the cosine
            similarity. If "count", the similarity is the number of common elements/tokens.
            Currently, the tokens are simply three-grams of alphanumeric characters.
            NOTE: If "count" is used, the underlying computations are significantly faster due to
            u16 integer arithmetic.

        apply_word_normalization (bool):
            Whether to apply word normalization to the columns before computing the similarity.
            The normalization is done by removing all non-alphanumeric characters and converting
            the string to lowercase. Decreases the pre-processing time but might lead to better
            results.
            Defaults to False.

        threads (int | None, optional):
            The number of threads to use for computation. Defaults to None (= number of physical cores).

        add_mapping (bool):
            Whether to add the row-col mapping of df_left and df_right to the result.
            Defaults to False.

        add_similarity (bool):
            Whether to add the similarity score to the result.
            Defaults to True.

        threading_dimension (Literal["left", "right", "auto"]):
            The dimension to parallelize.
            If "left", the left DataFrame is parallelized.
            If "right", the right DataFrame is parallelized.
            If "auto", we use left parallelization if the left DataFrame is significantly smaller
            than the right DataFrame.
            Defaults to "left".

    Returns:
        pl.DataFrame:
            A DataFrame containing the cosine similarity results of length
            less or equal to len(left) * top_n.
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
            if 100 * len(left) <= len(right):
                # left side is very small
                # hence it might be benefitial to parallelize the right side
                # comes at a cost of transposing the right slices thread-wise
                parallelize_left = False
            else:
                parallelize_left = True
        case _:
            raise ValueError(f"Threading {threading_dimension} not supported")

    match normalization:
        case "l2":
            normalize = True
        case "count":
            normalize = False
        case _:
            raise ValueError(f"Normalization {normalization} not supported")

    if apply_word_normalization:
        left = normalize_string_col(left, left_on)
        right = normalize_string_col(right, right_on)
        left_on = f"{left_on}_normalized"
        right_on = f"{right_on}_normalized"

    _map = awesome_cossim(
        left, right, left_on, right_on, top_n, threads, normalize, parallelize_left
    ).cast(
        {
            "row": pl.UInt32,
            "col": pl.UInt32,
            "sim": pl.Float32,
        }
    )

    return (
        left.with_row_index("row")
        .join(_map, on="row", how="left")
        .join(right.with_row_index("col"), on="col", suffix=suffix)
        .pipe(lambda df: df.drop("row", "col") if not add_mapping else df)
        .pipe(lambda df: df.drop("sim") if not add_similarity else df)
    )
