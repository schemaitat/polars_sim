import polars_sim as ps
import polars as pl

df_left = pl.DataFrame({
    "name": ["aaa", "aba"],
})

df_right = pl.DataFrame({
    "name": ["aaa", "aac"],
})

res = ps.join_sim(
    df_left,
    df_right,
    on="name",
    top_n=1,
    normalization="l2",
    add_mapping=True,
    threading_dimension="left",
    threads=1
)

print(df_left)
print(df_right)
print(res)