import asyncio

import polars as pl
import streamlit as st
from plotly import express as px
from src.scrape import PC_CODE_DICT, RESULTS_DATA_PATH, store_full_result

st.set_page_config("Election 2024", page_icon="🗳️", layout="centered")


INDIA = (
    "AAAP",
    "AIFB",
    "AITC",
    "BHRTADVSIP",
    "CPI",
    "CPI(M)",
    "CPI(ML)(L)",
    "DMK",
    "INC",
    "IUML",
    "JKN",
    "JKPDP",
    "JMM",
    "KEC",
    "KEC(M)",
    "MDMK",
    "NCPSP",
    "RJD",
    "RSP",
    "SHSUBT",
    "SP",
    "VCK",
    "VSIP",
)
NDA = (
    "ADAL",
    "AGP",
    "AJSUP",
    "AMMKMNKZ",
    "APTADMK",
    "BDJS",
    "BJP",
    "HAMS",
    "JD(S)",
    "JD(U)",
    "JnP",
    "LJPRV",
    "MNF",
    "NCP",
    "NDPP",
    "NPEP",
    "NPF",
    "PHJSP",
    "PMK",
    "RLD",
    "RLM",
    "RPI(A)",
    "RSPS",
    "SAD",
    "SBSP",
    "SHS",
    "SKM",
    "TDP",
    "TMC(M)",
    "UDP",
    "UPPL",
)


def preprocess(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        alliance=pl.when(pl.col("party").is_in(INDIA))
        .then(pl.lit("INDIA"))
        .when(pl.col("party").is_in(NDA))
        .then(pl.lit("NDA"))
        .otherwise(pl.lit("OTH")),
        state=pl.col("pc_code").replace(PC_CODE_DICT),
    )


async def fetch_current_result() -> None:
    if not RESULTS_DATA_PATH.exists():
        await store_full_result()
        st.balloons()

    if st.button("Fetch Current Result", type="primary", use_container_width=True):
        await store_full_result()
        st.balloons()


async def alliance_pie_chart(df: pl.LazyFrame):
    alliance_count = (
        (await df.collect_async())
        .get_column("alliance")
        .value_counts(sort=True)
        .with_columns(
            pl.col("alliance").add(" (").add(pl.col("count").cast(pl.String)).add(")")
        )
    )
    fig = px.pie(
        alliance_count,
        "alliance",
        "count",
        title="Alliance Seat Share Plot",
    )
    return fig


async def party_pie_chart(df: pl.LazyFrame, head: int = 10):
    party_count = (
        (await df.collect_async())
        .get_column("party")
        .value_counts(sort=True)
        .with_columns(
            pl.col("party").add(" (").add(pl.col("count").cast(pl.String)).add(")"),
        )
    )
    fig = px.pie(
        party_count.head(head),
        "party",
        "count",
        title="Party Seat Share Plot",
    )
    return fig


async def state_pie_chart(df: pl.LazyFrame, pc_code: str):
    state_count = await (
        df.filter(
            pl.col("pc_code").eq(pc_code),
        )
        .group_by("party")
        .agg(
            pl.first("state"),
            pl.len().alias("seats"),
        )
        .sort("seats", descending=True)
        .with_columns(
            pl.col("party").add(" (").add(pl.col("seats").cast(pl.String)).add(")"),
        )
        .collect_async()
    )
    fig = px.pie(
        state_count,
        "party",
        "seats",
        title=f"{PC_CODE_DICT[pc_code]} Seat Share Plot",
    )
    return fig


async def main():
    st.title("🇮🇳 India Election 2024 🇮🇳")
    await fetch_current_result()
    ldf = preprocess(pl.scan_csv(RESULTS_DATA_PATH))

    st.warning(
        "There is a error while choosing the alliance party. "
        "**That's why seats share is incorrect.** "
        "Please correct me if you got the solution.\n\n"
        "👀 See the **'Alliance Details'** expander component below.",
        icon="⚠️",
    )
    L, R = st.columns(2)
    alliance_fig = await alliance_pie_chart(ldf)
    party_fig = await party_pie_chart(ldf)
    L.plotly_chart(alliance_fig, use_container_width=True)
    R.plotly_chart(party_fig, use_container_width=True)

    with st.expander("🤝 **Alliance Details**"):
        st.write(f"**N.D.A:** `{NDA}`")
        st.write(f"**I.N.D.I.A:** `{INDIA}`")

    pc_code = st.selectbox(
        "Select State",
        PC_CODE_DICT,
        index=3,
        format_func=lambda x: PC_CODE_DICT[x],
    )
    if pc_code is None:
        st.error("Doesn't selected PC Code.")
        st.stop()
    state_fig = await state_pie_chart(ldf, pc_code)
    st.plotly_chart(state_fig, use_container_width=True)


if __name__ == "__main__":
    asyncio.run(main())
