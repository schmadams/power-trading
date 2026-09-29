import pandas as pd
import requests

from helpers.supabase_db import SupabaseDB
from helpers.check_env import check_env

check_env()

ELEXON_MID_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"

# Confirmed by testing directly against the live API: this endpoint hard
# caps each request's date range at 7 days -- exactly 7 days succeeds,
# 7 days + 1 second returns 400 Bad Request. Not documented anywhere, so
# don't push this higher.
CHUNK_DAYS = 7


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


def _fetch_mid_chunk(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    params = {
        "from": start.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": end.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    response = requests.get(ELEXON_MID_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json().get("data", [])

    if not data:
        return pd.DataFrame(columns=["startTime", "price", "volume"])

    df = pd.DataFrame(data)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    return df[["startTime", "price", "volume"]]


def get_gb_day_ahead_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch GB day-ahead reference prices from Elexon's Market Index Data (MID)
    endpoint and return a tidy DataFrame matching the auction_prices table
    schema: delivery_start, region, auction, price.

    ENTSO-E carries no GB day-ahead price data at all (GB exited SDAC after
    Brexit), so this hits Elexon's free, keyless API instead -- see
    get_day_ahead_prices in entsoe_da_prices.py for the other regions.

    Each settlement period has one price per data provider (currently
    APX/EPEX and N2EX). Since a provider can report zero volume for a given
    period, the price is volume-weighted across providers rather than
    picking one provider by name, so it keeps working if reporting patterns
    change.

    Example:
        start = pd.Timestamp("2026-07-01", tz="UTC")
        end = pd.Timestamp("2026-07-08", tz="UTC")
        df = get_gb_day_ahead_prices(start, end)
    """
    chunks = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS), end)
        chunks.append(_fetch_mid_chunk(chunk_start, chunk_end))
        chunk_start = chunk_end

    raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["startTime", "price", "volume"])

    if raw.empty:
        return pd.DataFrame(columns=["delivery_start", "region", "auction", "price"])

    raw["weighted"] = raw["price"] * raw["volume"]
    grouped = raw.groupby("startTime").agg(
        weighted_sum=("weighted", "sum"),
        volume_sum=("volume", "sum"),
        price_mean=("price", "mean"),
    ).reset_index()

    grouped["price"] = grouped.apply(
        lambda row: row["weighted_sum"] / row["volume_sum"] if row["volume_sum"] > 0 else row["price_mean"],
        axis=1,
    )

    result = grouped.rename(columns={"startTime": "delivery_start"})
    result["region"] = "GB"
    result["auction"] = "day-ahead"

    db_df =  result[["delivery_start", "region", "auction", "price"]]
    write_results_to_da_prices_table(db_df)


if __name__ == "__main__":
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp.now(tz="UTC")

    prices = get_gb_day_ahead_prices(start, end)
    write_results_to_da_prices_table(prices)
    print(prices)