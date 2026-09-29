from typing import Iterable

import pandas as pd
from entsoe import EntsoePandasClient
from helpers.check_env import check_env
from helpers.supabase_db import SupabaseDB

import os

check_env()

# entsoe-py area codes for the project's regions.
# Note: Germany must be queried as "DE_LU" (the DE-LU bidding zone since 2018),
# plain "DE" is not a valid ENTSO-E bidding zone code.
DEFAULT_REGIONS: list[str] = ["GB", "FR", "DE_LU", "BE", "NL", "DK_1", "DK_2"]

# Above this window width, treat the request as a backfill: write each
# region to Supabase as soon as it's fetched, rather than holding everything
# in memory and writing once at the end. Keeps a failure partway through
# from losing data already fetched, and avoids building a huge DataFrame
# in memory for multi-year pulls.
BACKFILL_THRESHOLD_DAYS = 30


def _fetch_region(client: EntsoePandasClient, region: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        series = client.query_day_ahead_prices(region, start=start, end=end)
    except Exception as exc:
        print(f"Failed to fetch day-ahead prices for {region}: {exc}")
        return pd.DataFrame(columns=["delivery_start", "region", "auction", "price"])

    df = series.rename("price").to_frame()
    df["region"] = region
    df["auction"] = "day-ahead"
    df["delivery_start"] = df.index
    return df.reset_index(drop=True)[["delivery_start", "region", "auction", "price"]]


def get_day_ahead_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    regions: Iterable[str] = DEFAULT_REGIONS,
) -> pd.DataFrame:
    """
    Fetch day-ahead auction prices from ENTSO-E for one or more regions and
    return a tidy DataFrame matching the auction_prices table schema:
    delivery_start, region, auction, price.

    If the requested window is wider than BACKFILL_THRESHOLD_DAYS, each
    region is written to Supabase as soon as it's fetched (see
    write_results_to_da_prices_table). For normal short windows (e.g. a
    daily poll), nothing is written here -- the caller writes the returned
    DataFrame once, as before.

    Example:
        start = pd.Timestamp("2026-07-01", tz="Europe/Brussels")
        end = pd.Timestamp("2026-07-08", tz="Europe/Brussels")
        df = get_day_ahead_prices(api_key, start, end)
    """

    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing ENTSOE_API_KEY environment variable.")
    client = EntsoePandasClient(api_key=api_key)
    is_backfill = (end - start).days > BACKFILL_THRESHOLD_DAYS

    frames = []
    for region in regions:
        df = _fetch_region(client, region, start, end)

        if is_backfill and not df.empty:
            write_results_to_da_prices_table(df)
            print(f"  Wrote {len(df)} rows for {region}.")

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["delivery_start", "region", "auction", "price"])

    db_df =  pd.concat(frames, ignore_index=True)
    if (end - start).days <= BACKFILL_THRESHOLD_DAYS:
        write_results_to_da_prices_table(db_df)


def write_results_to_da_prices_table(results: pd.DataFrame) -> None:
    """
    Upsert day-ahead price results into the auction_prices table,
    keyed on (delivery_start, region) to match the table's unique constraint.
    """
    if results.empty:
        print("No results to write.")
        return

    records = results.copy()
    records["delivery_start"] = records["delivery_start"].apply(lambda ts: ts.isoformat())
    records["price"] = records["price"].astype(float)

    db = SupabaseDB()
    db.write(
        "auction_prices",
        records.to_dict(orient="records"),
        upsert=True,
        on_conflict="delivery_start,region",
    )


if __name__ == "__main__":


    start = pd.Timestamp("2022-01-01", tz="Europe/Brussels")
    end = pd.Timestamp.now(tz="Europe/Brussels")

    prices = get_day_ahead_prices(start, end)

    # Backfills already write region-by-region inside get_day_ahead_prices,
    # so only write here for normal, narrow-window calls.
    if (end - start).days <= BACKFILL_THRESHOLD_DAYS:
        write_results_to_da_prices_table(prices)

    print(prices)