import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")

with app.setup:
    from datetime import datetime
    from urllib.parse import urlparse

    import altair as alt
    import marimo as mo
    import polars as pl
    import spacy
    import wordcloud
    from browser_history.generic import ChromiumBasedBrowser
    from matplotlib import pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load from History Database Directly
    """)


@app.cell
def _():
    # visits = pl.read_database_uri("SELECT * FROM visits", "sqlite://./data/History?mode=ro")
    # urls = pl.read_database_uri("SELECT * FROM urls", "sqlite://./data/History?mode=ro")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Using `browser-history` Package
    """)


@app.class_definition
class Helium(ChromiumBasedBrowser):
    name = "helium"
    history_file = "Developer/_rough/data/History"
    # mac_path = "/Users/iarv/Library/Application Support/net.imput.helium/Default"
    mac_path = ""


@app.cell
def _():
    # helium = Helium()
    # helium.fetch_history().save("data/history.csv")
    return


@app.function
def parse_url(url: str):
    parsed = urlparse(url)
    return {
        "netloc": parsed.netloc.removeprefix("www."),
        "path": parsed.path,
        "query": parsed.query,
    }


@app.cell
def _():
    history = (
        pl.scan_csv("data/history.csv")
        .rename(str.lower)
        .with_columns(
            pl.col("timestamp")
            .str.to_datetime("%Y-%m-%d %H:%M:%S%z", time_zone="Asia/Kolkata")
            .dt.replace_time_zone(None),
            pl.col("url").map_elements(parse_url).name.suffix("_parsed"),
        )
        .unnest("url_parsed")
    )
    return (history,)


@app.cell
def _(history):
    mo.ui.table(history.tail().collect())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Setup Spacy NLP
    """)


@app.cell
def _():
    nlp = spacy.load("en_core_web_sm")
    return (nlp,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Latest Week History
    """)


@app.cell
def _(history):
    week_history = history.filter(
        pl.col("timestamp").ge(datetime(2026, 2, 7)),
    ).collect()
    return (week_history,)


@app.cell(hide_code=True)
def _(week_history):
    mo.md(f"""
    The week from {week_history["timestamp"].dt.min():%Y-%m-%d} to {week_history["timestamp"].dt.max():%Y-%m-%d} has {week_history.height} history items.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Most Visited Websites
    """)


@app.function
def most_visited_sites_plot(df: pl.DataFrame, top_k: int = 10):
    return (
        df["netloc"]
        .value_counts()
        .top_k(top_k, by="count")
        .plot.bar(
            x=alt.X("netloc:N", sort="-y", title="Domain"),
            y=alt.Y("count:Q", title="Visit Count"),
        )
        .properties(
            title="Top 10 Domains by Visit Count",
        )
    )


@app.cell
def _(week_history):
    most_visited_sites_plot(week_history)


@app.function
def hourly_heatmap(df: pl.LazyFrame):
    return (
        df.group_by(
            hour=pl.col("timestamp").dt.hour(),
            day_of_week=pl.col("timestamp").dt.weekday(),
        )
        .len()
        .collect()
        .plot.rect(
            x=alt.X("hour:O", title="Hour of Day"),
            y=alt.Y("day_of_week:O", sort="-y", title="Day of Week"),
            color=alt.Color(
                "len:Q", scale=alt.Scale(scheme="viridis"), title="Activity"
            ),
        )
        .properties(
            title="Hourly Activity Heatmap",
            height=400,
        )
    )


@app.cell
def _(week_history):
    hourly_heatmap(week_history.lazy())


@app.function
def text_wordcloud(text: str, title: str = "Word Cloud"):
    cloud = wordcloud.WordCloud(
        width=800,
        height=400,
        max_words=100,
        collocations=False,
    ).generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


@app.cell
def _(week_history):
    # titles wordcloud
    _text = week_history.get_column("title").implode().list.join(" ").item()
    text_wordcloud(_text, title="Visited Website Titles Word Cloud")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    """)


@app.cell(hide_code=True)
def _():
    website_form = (
        mo.md("""
    ### Create Word Cloud from Visited Website Titles

    Website name: {website}

    Texts to ignore from titles _(comma separated values)_: {ignore_texts}

    > *Website name will be search through visited website URL's `netloc`.
    """)
        .batch(
            website=mo.ui.text(value="youtube.com"),
            ignore_texts=mo.ui.text(value="youtube,subscriptions", full_width=True),
        )
        .form()
    )
    website_form
    return (website_form,)


@app.cell
def _(nlp):
    def preprocess_text(text: str) -> str:
        processed = nlp(text)
        tokens = [t.lemma_ for t in processed if not t.is_stop and t.is_alpha]
        return " ".join(tokens)

    return (preprocess_text,)


@app.function
def get_wordcloud_text(
    history_df: pl.DataFrame,
    website: str,
    ignore_texts: list[str],
) -> str:
    return (
        history_df.filter(
            pl.col("netloc").str.contains(website, literal=True),
        )
        .get_column("title")
        .str.replace_many(
            ignore_texts + website.strip().split("."),
            [""],
            ascii_case_insensitive=True,
        )
        .implode()
        .list.join(" ")
        .item()
    )


@app.cell
def _(preprocess_text, website_form, week_history):
    mo.stop(
        not website_form.value,
        mo.callout("Fill the form first.", kind="danger"),
    )

    _text = get_wordcloud_text(
        week_history,
        website_form.value["website"],
        website_form.value["ignore_texts"].split(","),
    )
    text_wordcloud(preprocess_text(_text), title="YouTube Titles Word Cloud")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Browsing Volume

    No. of websites visited and no. of times visited each month or week.
    """)


@app.function
def monthly_visits_plot(df: pl.LazyFrame):
    return (
        df.group_by(pl.col("timestamp").dt.strftime("%m-%y"))
        .len()
        .sort("len", descending=True)
        .collect()
        .plot.bar(
            x=alt.X("timestamp", title="Timestamp"),
            y=alt.Y("len", title="Visit Count"),
        )
        .properties(
            title="Monthly Visit Count",
        )
    )


@app.cell
def _(history):
    # Filter out top 10 websites overy months.
    # Total visits over months.
    # def browsing_volume(df: pl.LazyFrame):
    # history.head().collect()

    (
        history
        # 1️⃣ Extract month (year-month)
        .with_columns(
            pl.col("timestamp").dt.month().alias("month"),
        )
        # 2️⃣ Count visits per month per netloc
        .group_by(["month", "netloc"])
        .agg(
            pl.len().alias("monthly_visits"),
        )
        # 3️⃣ Rank websites within each month
        .with_columns(
            pl.col("monthly_visits")
            .rank("dense", descending=True)
            .over("month")
            .alias("rank"),
        )
        # 4️⃣ Keep top 10 per month
        .filter(pl.col("rank") <= 10)
        # # 5️⃣ Now compute total visits across months
        # .group_by("netloc")
        # .agg(
        #     pl.sum("monthly_visits").alias("total_visits_across_months"),
        # )
        # # Optional: sort final output
        # .sort("total_visits_across_months", descending=True)
        .collect()
    )


@app.cell
def _(history):
    monthly_visits_plot(history)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Most Visited Paths by Website

    This section analyzes the most frequently visited paths for specific websites from your browsing history.
    We'll start with YouTube and GitHub, as they are commonly used, and then provide a way to check other websites.
    """)


@app.function
def plot_website_paths(
    history_df: pl.DataFrame,
    website_name: str,
    top_n: int = 10,
    path_categorizer=None,
    title_suffix: str = "",
):
    """
    Plots the most visited paths for a given website, with optional path categorization.

    Args:
        history_df: The DataFrame containing browser history.
        website_name: The domain name of the website.
        top_n: The number of top paths to display.
        path_categorizer: An optional function that takes a path string and returns
                          a categorized string. If None, raw paths are used.
        title_suffix: Additional text for the chart title.

    Returns:
        An Altair Chart object.
    """
    filtered_df = history_df.filter(
        pl.col("netloc").str.contains(website_name, literal=True),
    )

    if filtered_df.is_empty():
        return mo.md(
            f"No history found for **`{website_name}`** in the selected period.",
        )

    if path_categorizer:
        df_to_plot = (
            filtered_df.with_columns(
                pl.col("path").map_elements(path_categorizer).alias("categorized_path"),
            )
            .drop_nulls("categorized_path")
            .group_by("categorized_path")
            .len()
            .sort("len", descending=True)
            .head(top_n)
        )
        x_axis_label = "Categorized Path"
        x_axis_field = "categorized_path"
    else:
        df_to_plot = (
            filtered_df.group_by("path").len().sort("len", descending=True).head(top_n)
        )
        x_axis_label = "Path"
        x_axis_field = "path"

    if df_to_plot.is_empty():
        return mo.callout(
            f"No significant paths found for **`{website_name}`**.",
            kind="warn",
        )

    chart = df_to_plot.plot.bar(
        x=alt.X(x_axis_field + ":N", sort="-y", title=x_axis_label),
        y=alt.Y("len:Q", title="Visit Count"),
        tooltip=[
            alt.Tooltip(x_axis_field, title=x_axis_label),
            alt.Tooltip("len", title="Visits"),
        ],
    ).properties(
        title=f"Top {top_n} Most Visited Paths for {website_name} {title_suffix}",
    )
    return chart


@app.function
def github_path_categorizer(path: str) -> str | None:
    """Categorizes GitHub paths."""
    if path in ("", "/", "/feed"):
        return "GitHub Home/Feed"
    if path.startswith("/search"):
        return "Search Page"
    if path.count("/") == 1:
        return "User/Org Profile"
    if path.count("/") >= 2:
        parts = path.split("/")
        # Ensure user and repo parts exist
        if len(parts) >= 3 and parts[1] and parts[2]:
            return f"{parts[1]}/{parts[2]}"


@app.function
def youtube_path_categorizer(path: str) -> str | None:
    """Categorizes YouTube paths."""
    mapping = {
        "/watch": "Video Watch Page",
        ("/channel", "/user", "/@"): "Channel/User Page",
        "/playlist": "Playlist Page",
        "/feed/subscriptions": "Subscriptions Feed",
        "/results": "Search Results",
        "/shorts": "YouTube Shorts",
        "/live": "Live Stream",
    }

    if path == "/":
        return None  # "YouTube Home"
    for pattern, rt_value in mapping.values():
        if path.startswith(pattern):
            return rt_value
    return "Other YouTube Page"


@app.function
def folo_path_categorizer(path: str) -> str | None:
    """Categorizes YouTube paths."""
    if path == "/discover":
        return "Feed Discover"
    if path.startswith("/timeline/videos"):
        return "Videos Feed"
    if path.startswith("/timeline/articles"):
        return "Articles Feed"
    if path.startswith("/timeline/notifications"):
        return "Notifications Feed"
    if path.startswith("/timeline/social-media"):
        return "Social Media Feed"
    return "Other Folo.is Pages"


@app.cell
def _(website_form, week_history):
    mo.stop(
        not website_form.value,
        mo.callout("Please enter a website domain to analyze.", kind="danger"),
    )

    mo.ui.altair_chart(
        plot_website_paths(
            week_history,
            website_form.value["website"],
            path_categorizer=folo_path_categorizer,
        ),
    )


@app.function
def filter_by_time_window(
    history_df: pl.DataFrame,
    start_hour: int,
    end_hour: int,
) -> pl.DataFrame:
    """
    Filters a DataFrame to include only records within a specific time window (inclusive of start_hour, exclusive of end_hour).
    Handles time windows that span across midnight (e.g., 20-00).
    """
    hour_col = pl.col("timestamp").dt.hour()
    if start_hour < end_hour:
        # Standard time window within a single day
        return history_df.filter((hour_col >= start_hour) & (hour_col < end_hour))
    else:
        # Time window spans midnight (e.g., 20-00 means 20, 21, 22, 23)
        return history_df.filter((hour_col >= start_hour) | (hour_col < end_hour))


@app.function
def plot_top_sites_by_time_window(
    history_df: pl.DataFrame,
    start_hour: int,
    end_hour: int,
    title_suffix: str,
    top_n: int = 10,
):
    """
    Plots the top N most visited websites within a specified time window.
    """
    filtered_df = filter_by_time_window(history_df, start_hour, end_hour)

    if filtered_df.is_empty():
        return mo.callout(
            f"No history found for the time window {start_hour:02d}:00-{end_hour:02d}:00.",
            kind="warn",
        )

    df_to_plot = filtered_df["netloc"].value_counts().top_k(top_n, by="count")

    chart = df_to_plot.plot.bar(
        x=alt.X("netloc:N", sort="-y", title="Domain"),
        y=alt.Y("count:Q", title="Visit Count"),
        tooltip=[
            alt.Tooltip("netloc", title="Domain"),
            alt.Tooltip("count", title="Visits"),
        ],
    ).properties(
        title=f"Top {top_n} Most Visited Domains {title_suffix}",
    )
    return chart


@app.function
def plot_weekly_activity_by_time_window(
    history_df: pl.DataFrame,
    start_hour: int,
    end_hour: int,
    title_suffix: str,
):
    """
    Plots the weekly activity (visits per day of the week) within a specified time window.
    """
    filtered_df = filter_by_time_window(history_df, start_hour, end_hour)

    if filtered_df.is_empty():
        return mo.callout(
            f"No history found for the time window {start_hour:02d}:00-{end_hour:02d}:00.",
            kind="warn",
        )

    # Day of week: Monday=1, Sunday=7. We want to sort them correctly for visualization.
    # Convert to string and map for correct ordering in Altair
    df_to_plot = (
        filtered_df.group_by(
            day_of_week=pl.col("timestamp").dt.weekday().cast(pl.UInt8),
        )
        .len()
        .with_columns(
            pl.col("day_of_week")
            .replace_strict(
                {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"},
                return_dtype=pl.String,
            )
            .alias("day_name"),
        )
        .sort("day_of_week")
    )

    # Define the order for the days of the week
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    chart = df_to_plot.plot.bar(
        x=alt.X("day_name:O", sort=day_order, title="Day of Week"),
        y=alt.Y("len:Q", title="Visit Count"),
        tooltip=[
            alt.Tooltip("day_name", title="Day"),
            alt.Tooltip("len", title="Visits"),
        ],
    ).properties(
        title=f"Weekly Activity {title_suffix}",
    )
    return chart


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Late Night Browsing Analysis (20:00 - 00:00)
    """)


@app.cell
def _(week_history):
    late_night_start = 20
    late_night_end = 0  # This will represent up to 23:59:59

    _plot_1 = plot_top_sites_by_time_window(
        week_history,
        late_night_start,
        late_night_end,
        title_suffix=f"during Late Night ({late_night_start:02d}:00 - {late_night_end:02d}:00)",
    )
    _plot_2 = plot_weekly_activity_by_time_window(
        week_history,
        late_night_start,
        late_night_end,
        title_suffix=f"during Late Night ({late_night_start:02d}:00 - {late_night_end:02d}:00)",
    )
    mo.ui.altair_chart((_plot_1 & _plot_2), chart_selection=False)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Work Hours Browsing Analysis (10:00 - 17:00)
    """)


@app.cell
def _(week_history):
    work_hours_start = 10
    work_hours_end = 17

    _plot_1 = plot_top_sites_by_time_window(
        week_history,
        work_hours_start,
        work_hours_end,
        title_suffix=f"during Work Hours ({work_hours_start:02d}:00 - {work_hours_end:02d}:00)",
    )

    _plot_2 = plot_weekly_activity_by_time_window(
        week_history,
        work_hours_start,
        work_hours_end,
        title_suffix=f"during Work Hours ({work_hours_start:02d}:00 - {work_hours_end:02d}:00)",
    )

    mo.ui.altair_chart(_plot_1 & _plot_2)


if __name__ == "__main__":
    app.run()
