import pytest
import polars as pl
from polars.testing import assert_frame_equal
import polars_sim as ps


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            pl.DataFrame({"s": ["aaa"]}),
            pl.DataFrame({"s": ["aaa"]}),
            pl.DataFrame({"sim": [1], "row": [0], "col": [0]}),
        ),
        (
            pl.DataFrame({"s": ["aaabb"]}),
            pl.DataFrame({"s": ["aaa"]}),
            pl.DataFrame({"sim": [1 / 3**0.5], "row": [0], "col": [0]}),
        ),
        # check for symmetriy
        (
            pl.DataFrame({"s": ["aaa"]}),
            pl.DataFrame({"s": ["aaabb"]}),
            pl.DataFrame({"sim": [1 / 3**0.5], "row": [0], "col": [0]}),
        ),
        (
            # one matching token
            # right has 3 tokens
            pl.DataFrame({"s": ["abc"]}),
            pl.DataFrame({"s": ["abcabc"]}),
            pl.DataFrame({"sim": [1 / 3**0.5], "row": [0], "col": [0]}),
        ),
    ],
)
def test_join_sim_basic(left, right, expected):
    result = ps.join_sim(
        left,
        right,
        on="s",
        ntop=1,
        normalization="l2",
        threads=1,
        add_mapping=True,
        add_similarity=True,
        suffix="_right",
    )
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) == {"s", "s_right", "sim", "row", "col"}
    assert_frame_equal(
        result.drop("s", "s_right"),
        expected,
        check_column_order=False,
        check_dtypes=False,
        atol=1e-6,
    )
