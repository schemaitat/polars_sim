import marimo

__generated_with = "0.8.21"
app = marimo.App(width="medium")


@app.cell
def __():
    import polars_sim as ps
    import polars as pl
    return pl, ps


@app.cell
def __(pl):
    df = pl.DataFrame({
        "name" : ["test", "andre"]
    })
    return (df,)


@app.cell
def __(df, ps):
    ps.join_sim(df,df, on="name")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
