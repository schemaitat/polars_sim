import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import polars as pl
    import polars_sim as ps
    return mo, pl, ps


@app.cell
def __(pl):
    df = pl.DataFrame({
        "name" : ["andre", "andrea", "horst", "hor"]
    })
    return (df,)


@app.cell
def __(df, ps):
    ps.join_sim(
        df, df, on = "name", normalization="l2", ntop=2
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
