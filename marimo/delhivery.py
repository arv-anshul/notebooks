# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "polars[plot]==1.7.1",
# ]
# ///

import marimo

__generated_with = "0.8.18"
app = marimo.App(
    app_title="Case Study on Delhivery Dataset",
    width="full",
)


@app.cell
def __():
    from pathlib import Path

    import altair as alt
    import polars as pl

    import marimo as mo

    return Path, alt, mo, pl


@app.cell
def __(pl):
    # Set some config for notebook
    pl.Config.set_fmt_str_lengths(42).set_thousands_separator(True)
    None


@app.cell
def __(mo):
    mo.carousel(
        [
            mo.md("## About Delhivery"),
            mo.md("""
            Delhivery is the largest and fastest-growing fully integrated player in India by revenue in Fiscal 2021.

            They aim to build the operating system for commerce, through a combination of world-class infrastructure, logistics operations of the highest quality, and cutting-edge engineering and technology capabilities.
            """).center(),
            mo.md("## What is this data about?"),
            mo.md("""
            The data team at **Delhivery** builds intelligence and capabilities using this data that helps them to widen the gap between the quality, efficiency, and profitability of their business versus their competitors.
            """),
            mo.md("## How can you help here?"),
            mo.md("""
            The company wants to understand and process the data coming out of data engineering pipelines:

            - Clean, sanitize and manipulate data to get useful features out of raw fields.
            - Make sense out of the raw data and help the data science team to build forecasting models on it.
            """),
            mo.md("## Let's work on **Delhivery** data.").center(),
        ],
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Column Profiling

        - **`data`**: tells whether the data is testing or training data
        - **`trip_creation_time`**: Timestamp of trip creation
        - **`route_schedule_uuid`**: Unique Id for a particular route schedule
        - **`route_type`**: Transportation type
        - **`FTL`**: Full Truck Load: FTL shipments get to the destination sooner, as the truck is making no other pickups or drop-offs along the way
        - **`Carting`**: Handling system consisting of small vehicles (carts)
        - **`trip_uuid`**: Unique ID given to a particular trip (A trip may include different source and destination centers)
        - **`source_center`**: Source ID of trip origin
        - **`source_name`**: Source Name of trip origin
        - **`destination_cente`**: Destination ID
        - **`destination_name`**: Destination Name
        - **`od_start_time`**: Trip start time
        - **`od_end_time`**: Trip end time
        - **`start_scan_to_end_scan`**: Time taken to deliver from source to destination
        - **`is_cutoff`**: UNKNOWN
        - **`cutoff_factor`**: UNKNOWN
        - **`cutoff_timestamp`**: UNKNOWN
        - **`actual_distance_to_destination`**: Distance in Kms between source and destination warehouse
        - **`actual_time`**: Actual time taken to complete the delivery (Cumulative)
        - **`osrm_time`**: An open-source routing engine time calculator which computes the shortest path between points in a given map (Includes usual traffic, distance through major and minor roads) and gives the time (Cumulative)
        - **`osrm_distance`**: An open-source routing engine which computes the shortest path between points in a given map (Includes usual traffic, distance through major and minor roads) (Cumulative)
        - **`factor`**: UNKNOWN
        - **`segment_actual_time`**: This is a segment time. Time taken by the subset of the package delivery
        - **`segment_osrm_time`**: This is the OSRM segment time. Time taken by the subset of the package delivery
        - **`segment_osrm_distance`**: This is the OSRM distance. Distance covered by subset of the package delivery
        - **`segment_factor`**: UNKNOWN
        """,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.carousel(
        [
            mo.md("## Approach to Follow"),
            mo.md("""
        - Feature Creation
        - Relationship between Features
        - Column Normalization /Column Standardization
        - Handling categorical values
        - Missing values - Outlier treatment / Types of outliers
        """),
            mo.md("## How to begin"),
            mo.md("""
        Since delivery details of one package are divided into several rows _(think of it as connecting flights to reach a particular destination)_.

        - Now think about how we should treat their fields if we combine these rows?
        - What aggregation would make sense if we merge.
        - What would happen to the numeric fields if we merge the rows?
        """),
        ],
    )


@app.cell(hide_code=True)
def __(mo):
    _1 = mo.md("## Hints/Suggestions").center()
    _2 = mo.callout(
        mo.md("""
    You can use inbuilt functions like `group_by` and aggregations like `sum()`, `cumsum()` to merge rows based on their `trip_uuid`, `source_id` and `destination_id`, further aggregate on the basis of just `trip_uuid`.
    You can also keep the first and last values for some numerical or categorical fields if aggregating them won't make sense.
    """),
    )
    _3 = mo.accordion(
        {
            "**Basic data cleaning and exploration**": """
        - Handle missing values in the data.
        - Analyze the structure of the data.
        - Try merging the rows using the hint mentioned above.
        """,
            "**Build some features to prepare the data for actual analysis. Extract features from the below fields**": """
        - **Destination Name**: Split and extract features out of destination. City-place-code (State)
        - **Source Name**: Split and extract features out of destination. City-place-code (State)
        - **Trip_creation_time**: Extract features like month, year and day etc
        """,
            "**In-depth analysis and feature engineering**": """
        - Calculate the time taken between od_start_time and od_end_time and keep it as a feature. Drop the original columns, if required
        - Compare the difference between Point a. and start_scan_to_end_scan. Do hypothesis testing/ Visual analysis to check.
        - Do hypothesis testing/ visual analysis between actual_time aggregated value and OSRM time aggregated value (aggregated values are the values you'll get after merging the rows on the basis of trip_uuid)
        - Do hypothesis testing/ visual analysis between actual_time aggregated value and segment actual time aggregated value (aggregated values are the values you'll get after merging the rows on the basis of trip_uuid)
        - Do hypothesis testing/ visual analysis between osrm distance aggregated value and segment osrm distance aggregated value (aggregated values are the values you'll get after merging the rows on the basis of trip_uuid)
        - Do hypothesis testing/ visual analysis between osrm time aggregated value and segment osrm time aggregated value (aggregated values are the values you'll get after merging the rows on the basis of trip_uuid)
        - Find outliers in the numerical variables (you might find outliers in almost all the variables), and check it using visual analysis
        - Handle the outliers using the IQR method.
        - Do one-hot encoding of categorical variables (like route_type)
        - Normalize/Standardize the numerical features using MinMaxScaler or StandardScaler.
        """,
        },
    )
    mo.carousel([_1, _2, _3])


@app.cell
def __(pl):
    data_url = "https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/551/original/delhivery_data.csv"
    delhivery = pl.scan_csv(
        data_url,
        schema_overrides={
            "trip_creation_time": pl.Datetime,
            "od_start_time": pl.Datetime,
            "od_end_time": pl.Datetime,
            "cutoff_timestamp": pl.Datetime,
        },
    )
    return (delhivery,)


@app.cell(hide_code=True)
def __(delhivery, mo, pl):
    _df_height = delhivery.select("data").collect().height
    _null_count = (
        delhivery.null_count()
        .collect()
        .transpose(
            include_header=True,
            header_name="column",
            column_names=["null_count"],
        )
        .with_columns(
            null_pct=pl.col("null_count").truediv(_df_height),
        )
    )
    _unique_count = (
        delhivery.select(pl.all().n_unique())
        .collect()
        .transpose(
            include_header=True,
            header_name="column",
            column_names=["n_unique"],
        )
    )

    cat_cols = [
        "data",
        "route_type",
        "is_cutoff",
        "source_center",
        "source_name",
        "destination_center",
        "destination_name",
    ]
    datetime_cols = [
        "trip_creation_time",
        "od_start_time",
        "od_end_time",
        "cutoff_timestamp",
    ]
    id_cols = [
        "trip_uuid",
        "route_schedule_uuid",
    ]
    num_cols = [
        "start_scan_to_end_scan",
        "cutoff_factor",
        "actual_distance_to_destination",
        "actual_time",
        "osrm_time",
        "osrm_distance",
        "factor",
        "segment_actual_time",
        "segment_osrm_time",
        "segment_osrm_distance",
        "segment_factor",
    ]

    _1 = mo.md("## Basic overview of Dataset").center()
    _2 = mo.accordion(
        {
            "**Shape of Data**": mo.md(
                f"No. of Rows: **`{_df_height}`** <br/>"
                f"No. of Columns: **`{len(delhivery.collect_schema().names())}`**",
            ),
            "**Null Count of each column**": mo.vstack(
                [
                    mo.ui.table(_null_count, selection=None),
                    mo.md(
                        f"It seems there is total **`{_null_count['null_pct'].sum()}%`** null values in whole dataset.",
                    ).center(),
                ],
            ),
            "**Unique values per column**": mo.vstack(
                [
                    mo.ui.table(_unique_count, selection=None),
                ],
            ),
        },
        multiple=True,
        lazy=True,
    )

    mo.vstack([_1, _2])
    return cat_cols, datetime_cols, id_cols, num_cols


@app.cell(hide_code=True)
def __(cat_cols, delhivery, mo, num_cols):
    mo.vstack(
        [
            mo.md("### Categorical Columns").center(),
            mo.ui.table(
                delhivery.select(cat_cols).head(10).collect(),
                label="Overview Only",
                selection=None,
                show_column_summaries=False,
            ),
            mo.md("### Numerical Columns").center(),
            mo.ui.table(
                delhivery.select(num_cols).head(10).collect(),
                label="Overview Only",
                selection=None,
                show_column_summaries=False,
            ),
        ],
    )


@app.cell
def __(alt, cat_cols, delhivery, mo, pl):
    def _cat_cols_countplot(df, cat_cols):
        charts = []  # list to hold each chart

        # Create a chart for each categorical column
        for col in cat_cols:
            chart = (
                df.select(pl.col(col).value_counts())
                .collect()
                .unnest(col)
                .plot.bar(
                    x=alt.X(col, type="nominal", title=col),
                    y=alt.Y("count", title="Count"),
                )
                .properties(width=300, height=250)
            )
            charts.append(chart)

        # Combine the charts horizontally
        final_chart = (
            alt.hconcat(*charts)
            .resolve_scale(y="shared")
            .configure_axis(labelFontSize=12, titleFontSize=14)
            .configure_title(fontSize=16)
        )
        return final_chart

    mo.vstack(
        [
            mo.md("## Categorical Columns CountPlot").center(),
            mo.ui.altair_chart(
                _cat_cols_countplot(delhivery, cat_cols),
                chart_selection=False,
            ),
        ],
    )


@app.cell
def __(mo):
    bin_count = mo.ui.slider(2, 20, 1, 10, label="Select Bin Count")
    return (bin_count,)


@app.cell
def __(alt, bin_count, delhivery, mo, num_cols, pl):
    def _num_cols_distribution(df: pl.DataFrame, num_cols, *, bin_count: int = 10):
        charts = []
        for col in num_cols:
            chart = (
                df.select(
                    pl.col(col).hist(bin_count=bin_count, include_breakpoint=True),
                )
                .collect()
                .unnest(col)
                .rename({"breakpoint": "bins"})
                .plot.bar(x=alt.X("bins", bin=True, title=col), y="count")
            )
            charts.append(chart)

        final_chart = (
            alt.hconcat(*charts)
            .resolve_scale(y="shared")
            .configure_axis(labelFontSize=12, titleFontSize=14)
            .configure_title(fontSize=16)
        )
        return final_chart

    mo.vstack(
        [
            mo.md("## Numerical Column HistPlot").center(),
            bin_count,
            mo.ui.altair_chart(
                _num_cols_distribution(delhivery, num_cols, bin_count=bin_count.value),
                chart_selection=False,
            ),
        ],
    )


@app.cell
def __(alt, bin_count, delhivery, mo, num_cols, pl):
    def _num_cols_distribution(df: pl.DataFrame, num_cols, *, bin_count: int = 10):
        charts = []
        for col in num_cols:
            chart = (
                df.select(
                    pl.col(col).hist(bin_count=bin_count, include_breakpoint=True),
                )
                .collect()
                .unnest(col)
                .rename({"breakpoint": "bins"})
                .plot.bar(x=alt.X("bins", bin=True, title=col), y="count")
            )
            charts.append(chart)

        return charts

    _charts = _num_cols_distribution(delhivery, num_cols, bin_count=bin_count.value)
    mo.vstack(
        [
            mo.md("## Numerical Column HistPlot").center(),
            bin_count.center(),
            mo.carousel(
                mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)
                for chart in _charts
            ),
        ],
    )


if __name__ == "__main__":
    app.run()