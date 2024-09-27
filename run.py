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
    normalize=True,
)

print(df)
