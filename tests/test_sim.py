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
        (
            # left has one row
            # right has 2 rows
            pl.DataFrame({"s": ["abc", "def"]}),
            pl.DataFrame({"s": ["abc", "aaa"]}),
            pl.DataFrame({"sim": [1], "row": [0], "col": [0]}),
        ),
    ],
)
def test_join_sim_basic(left, right, expected):
    kwargs_to_test = [
        {"threads": 1, "threading_dimension": "left"},
        {"threads": 2, "threading_dimension": "left"},
        {"threading_dimension": "auto"},
    ]

    for kw in kwargs_to_test:
        result = ps.join_sim(
            left,
            right,
            on="s",
            top_n=1,
            normalization="l2",
            add_mapping=True,
            add_similarity=True,
            suffix="_right",
            **kw,
        )
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"s", "s_right", "sim", "row", "col"}
        assert_frame_equal(
            result.drop("s", "s_right"),
            expected,
            check_column_order=False,
            check_dtypes=False,
            atol=1e-31,
        )
