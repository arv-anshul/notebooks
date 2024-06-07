"""
This is a marimo notebook. Marimo is an open-source reactive notebook for Python.

Marimo Website: https://marimo.io
"""

import marimo

__generated_with = "0.6.15"
app = marimo.App(width="medium", app_title="SQL with Polars")


@app.cell
def __():
    import polars as pl

    import marimo as mo

    return mo, pl


@app.cell
def __(pl):
    df = pl.read_csv(
        "https://github.com/arv-anshul/notebooks/raw/main/data/data_job_salaries.csv"
    )
    print(df.shape)
    df.head()
    return (df,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # `pl.SQLContext`

        In this [marimo] notebook I am going to learn a new polars feature i.e. to run vanilla **SQL** queries with `polars.DataFrame`.

        Checkout the [docs](https://docs.pola.rs/user-guide/sql/intro/) for `pl.SQLContext`.

        [marimo]: https://marimo.io
        """
    )
    return


@app.cell
def __(df, pl):
    ctx = pl.SQLContext(df=df, eager=True)  # Register the DataFrame as `df`
    return (ctx,)


@app.cell
def __(mo):
    mo.md(r"### Top 5 rows of dataset")
    return


@app.cell
def __(ctx):
    ctx.execute("SELECT * from df LIMIT 5")
    return


@app.cell
def __(mo):
    mo.md(r"### Q. Calculate average salary for each expereience level.")
    return


@app.cell
def __(ctx):
    ctx.execute(
        "SELECT experience_level, avg(salary_in_usd) FROM df GROUP BY experience_level"
    )
    return


@app.cell
def __(mo):
    mo.md(r"### Q. Average salary different Jobs in each country for different year.")
    return


@app.cell
def __(ctx, mo):
    country = mo.ui.dropdown(
        ctx.execute("SELECT distinct company_location FROM df")
        .get_column("company_location")
        .to_list(),
        "IN",
        label="Select Country",
        allow_select_none=False,
    )
    country
    return (country,)


@app.cell
def __(ctx, mo, pl):
    year = mo.ui.dropdown(
        ctx.execute("SELECT distinct work_year FROM df")
        .get_column("work_year")
        .cast(pl.String)
        .to_list(),
        "2024",
        label="Select Year",
    )
    year
    return (year,)


@app.cell
def __(country, ctx, year):
    # ctx.execute(
    #     f"""
    #     SELECT work_year, company_location, avg(salary_in_usd) FROM df
    #     FILTER BY work_year={year.value} and company_location={country.value}
    #     GROUP BY work_year, company_location
    #     ORDER BY company_location, work_year
    #     """
    # )
    return


@app.cell
def __(country, df, pl):
    df.select(pl.col("company_location").eq(country.value).sum())
    return


if __name__ == "__main__":
    app.run()
