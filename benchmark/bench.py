import marimo

__generated_with = "0.8.21"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import time

    import polars as pl
    import polars_sim as ps
    from faker import Faker

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    return Faker, TfidfVectorizer, pl, ps, sp_matmul_topn, time


@app.cell
def __(Faker):
    fake = Faker()
    fake.seed_instance(4321)
    names_small = [fake.name() for _ in range(500)]
    names_big = [fake.name() for _ in range(100_000)]
    return fake, names_big, names_small


@app.cell
def __(names_big, names_small, pl):
    df_left = pl.DataFrame({"name": names_small})

    df_right = pl.DataFrame({"name": names_big})
    return df_left, df_right


@app.cell
def __(df_left, df_right, ps, time):
    def benchmark_topn():
        times = []
        for i in range(1, 10):
            start = time.time()
            ps.join_sim(
                df_left,
                df_right,
                on="name",
                ntop=i,
                normalization="l2",
            )
            end = time.time()
            times.append(end - start)
        return times

    return (benchmark_topn,)


@app.cell
def __(benchmark_topn):
    ntop_times = benchmark_topn()
    return (ntop_times,)


@app.cell
def __(ntop_times):
    ntop_times
    return


@app.cell
def __(df_left, df_right, ps):
    ps.join_sim(
        df_left,
        df_right,
        on="name",
        ntop=1,
        normalization="l2",
    ).sort("sim").tail(20)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
