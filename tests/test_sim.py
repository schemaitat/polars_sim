import polars as pl
from polars.testing import assert_frame_equal
import polars_sim as ps


def test_join_sim_basic():
    left = pl.DataFrame({"s": ["aaa", "aabbb", "abc"]})
    right = pl.DataFrame({"t": ["aaa", "aab", "def"]})

    result = ps.join_sim(
        left,
        right,
        left_on="s",
        right_on="t",
        ntop=1,
        normalization="count",
        threads=1,
        add_mapping=True,
        add_similarity=True,
    )
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) == {"s", "t", "sim", "row", "col"}
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "s": ["aaa", "aabbb"],
                "t": ["aaa", "aab"],
                "sim": [1.0, 1],
                "row": [0, 1],
                "col": [0, 1],
            }
        ),
        check_column_order=False,
        check_dtypes=False,
    )
