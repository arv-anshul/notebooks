import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import polars as pl

BASE_URL = "https://results.eci.gov.in/PcResultGenJune2024"
DATA_PATH = Path("data")
PC_CODE_DICT = json.loads((DATA_PATH / "pc_code.json").read_bytes())
GEOMETRY_DATA_PATH = DATA_PATH / "geometry.json"
RESULTS_DATA_PATH = DATA_PATH / "results.csv"


async def get_statewise_result(
    client: httpx.AsyncClient,
    state_id: str,
) -> dict[str, Any]:
    # TODO: Remove print statement
    print(f"Fetching: {state_id}")
    res = await client.get(f"/election-json-{state_id}-live.json")
    if res.status_code != 200:
        msg = f"Bad status code [{res.status_code}] for {BASE_URL!r}"
        raise RuntimeError(msg)
    return res.json()[state_id]


async def load_statewise_data(data: dict[str, Any]) -> pl.LazyFrame:
    chart_df = pl.LazyFrame(
        data["chartData"],
        schema=["party", "pc_code", "pc_no", "candidate", "hex"],
    )
    table_df = pl.LazyFrame(
        data["tableData"],
        schema=["party", "pc_code", "pc_no"],
    )
    return chart_df.join(
        table_df,
        on=["party", "pc_code", "pc_no"],
        validate="1:1",
    )


async def fetch_all_state_result(client: httpx.AsyncClient) -> list[pl.LazyFrame]:
    tasks = [get_statewise_result(client, pc_code) for pc_code in PC_CODE_DICT]
    return await asyncio.gather(
        *[load_statewise_data(data) for data in await asyncio.gather(*tasks)]
    )


async def get_statewise_geometry(
    client: httpx.AsyncClient,
    state_id: str,
) -> list[dict[str, Any]]:
    # TODO: Remove print statement
    print(f"Fetching: {state_id}")
    res = await client.get(f"/pc/{state_id}.js")
    if res.status_code != 200:
        msg = f"Bad status code [{res.status_code}] for {BASE_URL!r}"
        raise RuntimeError(msg)
    return json.loads(res.text[17:])


async def fetch_all_state_geometry(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    return [
        j
        for i in await asyncio.gather(
            *[get_statewise_geometry(client, pc_code) for pc_code in PC_CODE_DICT]
        )
        for j in i
    ]


async def store_full_result(client: httpx.AsyncClient | None = None):
    async with client or httpx.AsyncClient(base_url=BASE_URL) as client:
        full_result = await fetch_all_state_result(client)
    ldfs = await pl.concat(full_result).collect_async()
    RESULTS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    ldfs.write_csv(RESULTS_DATA_PATH)


async def store_full_geometry(client: httpx.AsyncClient | None = None):
    async with client or httpx.AsyncClient(base_url=BASE_URL) as client:
        full_geometry = await fetch_all_state_geometry(client)
    GEOMETRY_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GEOMETRY_DATA_PATH.open("w") as f:
        json.dump(full_geometry, f)


async def main():
    # await store_full_geometry()
    await store_full_result()


if __name__ == "__main__":
    asyncio.run(main())
