import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium", app_title="Benchmark", auto_download=["html"])


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md("""# Benchmark""")
    return


@app.cell
def __():
    import time

    from multiprocessing import cpu_count

    import polars as pl
    import polars_simed as ps

    from faker import Faker

    import plotly.express as px

    return Faker, cpu_count, pl, ps, px, time


@app.cell
def __(mo):
    def get_arg(name: str, type, default):
        arg_val = mo.cli_args().get(name)
        if arg_val is None:
            return default
        else:
            return type(arg_val)

    return (get_arg,)


@app.cell
def __(get_arg):
    size_left = get_arg("size_left", int, 5000)
    size_right = get_arg("size_right", int, 10_000)
    size_left, size_right
    return size_left, size_right


@app.cell
def __(pl):
    def append_row(df: pl.DataFrame, data):
        """
        Append a row to a polars dataframe.
        """
        return pl.concat([df, pl.DataFrame(data)], how="vertical_relaxed")

    return (append_row,)


@app.cell
def __(Faker, size_left, size_right):
    fake = Faker()
    fake.seed_instance(4321)
    names_small = [fake.name() for _ in range(size_left)]
    names_big = [fake.name() for _ in range(size_right)]
    return fake, names_big, names_small


@app.cell
def __(names_big, names_small, pl):
    df_left = pl.DataFrame({"name": names_small})
    df_right = pl.DataFrame({"name": names_big})
    return df_left, df_right


@app.cell
def __(Iterable, append_row, pl, ps, time):
    def benchmark(
        df_left: pl.DataFrame,
        df_right: pl.DataFrame,
        argument_name: str,
        argument_values: Iterable,
        value_dtype,
        **kwargs,
    ) -> pl.DataFrame:
        if kwargs is None:
            kwargs = {}

        df_bench = pl.DataFrame(
            schema={
                "argument_name": pl.Utf8,
                "argument_value": value_dtype,
                "time": pl.Float32,
                "size_left": pl.Int32,
                "size_right": pl.Int32,
            }
        )

        for val in argument_values:
            kwargs.update(
                {
                    argument_name: val,
                }
            )

            start = time.time()
            ps.join_sim(df_left, df_right, **kwargs)
            end = time.time()

            elapsed_time = end - start

            df_bench = append_row(
                df_bench,
                [
                    {
                        "argument_name": argument_name,
                        "argument_value": val,
                        "time": elapsed_time,
                        "size_left": len(df_left),
                        "size_right": len(df_right),
                    }
                ],
            )

        return df_bench

    return (benchmark,)


@app.cell
def __(px):
    def plot_benchmark(df_bench):
        argument_name = df_bench["argument_name"].unique()[0]
        size_left = df_bench["size_left"].unique()[0]
        size_right = df_bench["size_right"].unique()[0]

        fig = px.bar(
            df_bench,
            x="argument_value",
            y="time",
            title=f"Execution time for different {argument_name} values.<br>Dataframe dimensions: ({size_left},{size_right}).",
        )

        return fig

    return (plot_benchmark,)


@app.cell
def __(cpu_count, df_left, df_right, pl):
    benchmarks = {
        "ntop": {
            "df_left": df_left,
            "df_right": df_right,
            "argument_name": "top_n",
            "argument_values": range(1, 100, 10),
            "value_dtype": pl.Int32,
        },
        "threads": {
            "df_left": df_left,
            "df_right": df_right,
            "argument_name": "threads",
            "argument_values": range(1, cpu_count() + 1),
            "value_dtype": pl.Int32,
        },
        "normalizatåion": {
            "df_left": df_left,
            "df_right": df_right,
            "argument_name": "normalization",
            "argument_values": ["l2", "count"],
            "value_dtype": pl.Utf8,
        },
        "threading_coordinate": {
            "df_left": df_left,
            "df_right": df_right,
            "argument_name": "threading_dimension",
            "argument_values": ["left", "right"],
            "value_dtype": pl.Utf8,
        },
    }
    return (benchmarks,)


@app.cell
def __(benchmark, benchmarks, plot_benchmark):
    figs = {}
    for name, kw in benchmarks.items():
        df_l = benchmark(on="name", **kw, threading_dimension="left")
        df_r = benchmark(on="name", **kw, threading_dimension="right")

        figs[f"{name}_left"] = plot_benchmark(df_l)
        figs[f"{name}_right"] = plot_benchmark(df_r)
    return df_l, df_r, figs, kw, name


@app.cell
def __(figs):
    figs
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
