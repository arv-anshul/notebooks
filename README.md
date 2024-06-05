# Notebooks

Contains all my 📓 Notebooks where I have performed Data Analysis on unique datasets.

> \[!IMPORTANT\]
>
> See [`data/README.md`](data/README.md) to know about datasets I have used.

## Directory Info

### [`yt-watch-history`](yt-watch-history/)

Perform analysis on **YouTube Watch History** data _(exported via Google Takeout)_.

- > Previously using Pandas but switched to Polars as I started exploring it.
- Used `polars`'s amazing syntax to handle data, preprocess the text data and handle datetime data.
- Plot many graphs to show some amazing insights present in data.
- Build ML model to predict videos **"Content Type"** from its title.
- Build a **Channel Recommender System** which recommends similar channels from channel's videos' title and tags.

### [`spotify-analysis`](spotify-analysis/)

Perform analysis on **Spotify Streaming History** data _(exported via Spotify website)_.

- Analysed data from the perspective of Track, Artist, Album, Playlist and Time.
- Used `polars` builtin `plot` namespace (which uses `hvplot` library internally) to plot analysis graphs.

### [`credit-modeling`](credit-modeling/)

A project from **CampusX's free course on Credit Risk Modeling by Rohan Azad**.

- Collaborated with [@sambhavm22].
- Perform data analysis, build ML model using diffrent ML algorithms.
- Contains notebooks of mine and [@sambhavm22] both.
- Got many insights about banking sector.
- Created diagrams to explain the project workflows.
- [Credit Risk Modeling project documentation] in PDF format.

[Credit Risk Modeling project documentation]: credit-modeling/docs/DOCUMENTATION.pdf
[@sambhavm22]: https://github.com/sambhavm22

### [`election-2024`](election-2024/)

Created a dashboard using Streamlit which fetches data from ECI official website.

- Used `httpx` to fetch data asynchronously.
- Used `polars.LazyFrame` to manipulate data efficiently.
- Used `streamlit` to create dashboard.

<details>
<summary>Where is Notebooks?</summary>

There are no notebooks present in this project because I've converted those into `.py`
scripts because I have to create a dashboard using it and converted notebook's
non-`async` codes into `async` code.

</details>

### [`extra`](extra/)

This directory contains extra notebooks which are independent of each others.
Created these notebooks just for learning or fun purpose.
