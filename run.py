import polars_sim as ac
import polars as pl

df_left = pl.DataFrame({"names" : ["John", "Doe", "Jane", "Doe", "John", "Jane"]})
df_right = pl.DataFrame({"names" : ["John", "Doe", "Jane", "Doe", "John", "Jane"]})

# df_left = pl.concat([df_left for _ in range(100_000)])
# df_right = pl.concat([df_right for _ in range(10_000)])

s = ac.join_sim(
    df_left, 
    df_right, 
    on="names",
    ntop=10,
)

print(s)