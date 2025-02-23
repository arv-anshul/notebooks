# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "polars==1.22.0",
# ]
# ///

import marimo

__generated_with = "0.11.8"
app = marimo.App(width="full")


@app.cell
def _():
    import glob

    import marimo as mo
    import polars as pl
    from polars import selectors as cs

    return cs, glob, mo, pl


@app.cell
def _(pl):
    _streaming_reason = [
        "unexpected-exit-while-paused",
        "clickrow",
        "trackdone",
        "playbtn",
        "unknown",
        "trackerror",
        "appload",
        "endplay",
        "logout",
        "remote",
        "fwdbtn",
        "unexpected-exit",
        "backbtn",
    ]

    schema = {
        "ts": pl.Datetime,
        "platform": pl.String,
        "ms_played": pl.UInt64,
        "conn_country": pl.String,  # pl.Categorical,
        # "ip_addr": pl.String,
        "master_metadata_track_name": pl.String,
        "master_metadata_album_artist_name": pl.String,
        "master_metadata_album_album_name": pl.String,
        "spotify_track_uri": pl.String,
        # "episode_name": pl.String,
        # "episode_show_name": pl.String,
        # "spotify_episode_uri": pl.String,
        # "audiobook_title": pl.String,
        # "audiobook_uri": pl.String,
        # "audiobook_chapter_uri": pl.String,
        # "audiobook_chapter_title": pl.String,
        "reason_start": pl.String,  # pl.Enum(_streaming_reason),
        "reason_end": pl.String,  # pl.Enum(_streaming_reason),
        "shuffle": pl.Boolean,
        "skipped": pl.Boolean,
        "offline": pl.Boolean,
        # "offline_timestamp": pl.Time,
        "incognito_mode": pl.Boolean,
    }
    return (schema,)


@app.cell
def _(glob, pl, schema):
    _rename_cols = {
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist_name",
        "master_metadata_album_album_name": "album_name",
        "spotify_track_uri": "track_uri",
    }

    data = pl.concat(
        pl.read_json(
            path,
            schema=schema,
        ).rename(_rename_cols)
        for path in glob.glob("spotify-data/*.json")
    ).lazy()
    data.head(10).collect()
    return (data,)


@app.cell
def _(mo):
    mo.md("""## Null info""")


@app.cell
def _(data):
    data.null_count().collect()


@app.cell
def _(data, pl):
    _cols = ["track_name", "artist_name", "album_name", "track_uri"]
    data.filter(pl.col(c).is_null() for c in _cols).collect()


@app.cell
def _(mo):
    mo.md(r"""## EDA on data""")


@app.cell
def _(mo):
    mo.md(r"""### Different platforms I have used to listen music.""")


@app.cell
def _(data, pl):
    data.select(pl.col("platform").unique()).collect()["platform"].to_list()


@app.cell
def _(data, pl):
    (
        data.select(pl.col("platform").value_counts(normalize=True, sort=True))
        .unnest("platform")
        .with_columns(pl.col("proportion").mul(100).round(2))
        .collect()
    )


@app.cell
def _(mo):
    mo.md(
        r"""We can even narrow down the platform by **removing the version of browsers and OS**."""
    )


@app.cell
def _(mo):
    mo.md("""### Different country""")


@app.cell
def _(data, pl):
    data.select(pl.col("conn_country")).collect()["conn_country"].value_counts()


@app.cell
def _(data, pl):
    data.filter(pl.col("conn_country").ne("IN")).collect()


@app.cell
def _(mo):
    mo.md(r"""### Yearly streaming info""")


@app.cell
def _(data, mo, pl):
    _year_range = (
        f"Date rage of dataset: **{data.select(pl.col('ts').dt.date().min()).collect().item()}** "
        f"TO **{data.select(pl.col('ts').dt.date().max()).collect().item()}**"
    )

    _each_year_count = (
        data.group_by(year=pl.col("ts").dt.year())
        .agg(
            streaming_count=pl.len(),
            track_count=pl.col("track_uri").n_unique(),
            album_count=pl.col("album_name").n_unique(),
            artist_count=pl.col("artist_name").n_unique(),
        )
        .sort("year")
        .collect()
    )

    mo.vstack(
        [
            mo.md(_year_range),
            mo.as_html(_each_year_count),
        ],
    )


@app.cell
def _(mo):
    mo.md(r"""## Artists Info""")


@app.cell
def _(mo, sl_artist):
    mo.md(rf"""### When do **{sl_artist.value!r}** first played in the dataset.""")


@app.cell
def _(mo):
    sl_artist = mo.ui.text("Bharat Chauhan", "Artist Name")
    sl_artist
    return (sl_artist,)


@app.cell
def _(data, mo, pl, sl_artist):
    # Validate if entered sl_artist is in the dataset
    _artist_available = (
        data.filter(pl.col("artist_name").eq(sl_artist.value)).collect().is_empty()
    )
    mo.stop(
        _artist_available,
        mo.callout(mo.md(f"Artist **{sl_artist.value!r}** is not in data."), "danger"),
    )


@app.cell
def _(data, pl, sl_artist):
    (
        data.filter(pl.col("artist_name").eq(sl_artist.value))
        .filter(pl.col("ts").eq(pl.col("ts").min()))
        .collect()
    )


@app.cell
def _(mo, sl_artist):
    mo.md(rf"""### Yearly streaming of **{sl_artist.value!r}**""")


@app.cell
def _(data, pl, sl_artist):
    (
        data.filter(pl.col("artist_name").eq(sl_artist.value))
        .group_by(year=pl.col("ts").dt.year())
        .agg(
            streaming_count=pl.len(),
            track_count=pl.col("track_uri").n_unique(),
            album_count=pl.col("album_name").n_unique(),
        )
        .sort("year")
        .collect()
    )


@app.cell
def _(mo, sl_artist):
    mo.md(rf"""### All tracks of **{sl_artist.value!r}** with some details attached.""")


@app.cell
def _(data, mo, pl, sl_artist):
    mo.ui.table(
        data.filter(pl.col("artist_name").eq(sl_artist.value))
        .group_by("track_name")
        .agg(
            pl.col("artist_name").first(),
            streaming_count=pl.len(),
            first_stream=pl.col("ts").min().dt.date(),
        )
        .sort("first_stream", "streaming_count", descending=[False, True])
        .collect()
    )


@app.cell
def _(mo):
    mo.md(
        "I can also include `track_uri` column in `.group_by()` function by didn't because "
        "it segregate many track for no reason."
    )


@app.cell
def _(mo):
    mo.md(r"""## Export the data in `ndjson`/`jsonl` format.""")


@app.cell
def _(data):
    data.sink_csv("spotify-data/streaming_history.csv")


if __name__ == "__main__":
    app.run()
