import os

import pandas as pd
import requests

from helpers.check_env import check_env
from helpers.supabase_db import SupabaseDB

check_env()

# Confirmed against a live response: getauctions returns one row per
# corridor per day, with a nested "results" list holding 24 hourly product
# entries (productHour "15:00-16:00" etc), each with its own auctionPrice
# and capacity figures. marketPeriodStart is the UTC instant for the start
# of that delivery day.

JAO_BASE_URL = "https://api.jao.eu/OWSMP"

# Confirmed against a live call to get_corridors() -- GB-FR is not a single
# corridor, it's three separate cables each auctioned independently (IFA1,
# IFA2, ElecLink), and each named interconnector uses its own prefix code
# rather than a plain "COUNTRY-COUNTRY" pair. Denmark's zones show up as
# D1/D2 in JAO's naming, matching DK_1/DK_2.
#
# Capacity auctions are directional -- GB->FR and FR->GB are separate
# products, usually priced separately since congestion isn't symmetric.
# Each interconnector below lists both directions; both need fetching to
# get the full picture on that link.
CORRIDORS: dict[str, tuple[str, str]] = {
    "IFA1": ("IF1-GB-FR", "IF1-FR-GB"),
    "IFA2": ("IF2-GB-FR", "IF2-FR-GB"),
    "ELECLINK": ("EL1-GB-FR", "EL1-FR-GB"),
    "NEMO": ("NLL-GB-BE", "NLL-BE-GB"),
    "BRITNED": ("BDL-GB-NL", "BDL-NL-GB"),
    "VIKING_LINK": ("VKL-GB-D1", "VKL-D1-GB"),
    "DE_CH": ("DE-CH", "CH-DE"),
}

# JAO caps each getauctions request at a 31-day window.
CHUNK_DAYS = 31


def _headers() -> dict[str, str]:
    api_key = os.getenv("JAO_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing JAO_API_KEY environment variable.")
    return {"AUTH_API_KEY": api_key}


def get_corridors() -> list[str]:
    """
    List every corridor code JAO knows about. Use this to confirm the exact
    codes for IFA, Nemo Link, BritNed, Viking Link, and DE-CH before relying
    on the CORRIDORS dict above.
    """
    response = requests.get(f"{JAO_BASE_URL}/getcorridors", headers=_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def get_horizons() -> list[str]:
    """List every horizon name JAO knows about (Yearly, Monthly, Daily, etc.)."""
    response = requests.get(f"{JAO_BASE_URL}/gethorizons", headers=_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def _fetch_auctions_chunk(corridor: str, start: pd.Timestamp, end: pd.Timestamp, horizon: str) -> pd.DataFrame:
    params = {
        "corridor": corridor,
        "horizon": horizon,
        "fromdate": start.strftime("%Y-%m-%d-%H:%M:%S"),
        "todate": end.strftime("%Y-%m-%d-%H:%M:%S"),
    }

    try:
        response = requests.get(f"{JAO_BASE_URL}/getauctions", headers=_headers(), params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        # Don't let one corridor's failure (e.g. a shadow-auction-only
        # window, or a corridor with no data for this range) kill the whole
        # run -- log it and move on, same pattern as the GB ENTSO-E case.
        print(f"  Failed to fetch {corridor} for {start.date()} to {end.date()}: {exc}")
        return pd.DataFrame()

    data = response.json()

    if not data:
        return pd.DataFrame()

    # Flatten whatever JSON structure comes back rather than assuming field
    # names -- I couldn't confirm the exact response schema without a live
    # token/GitHub access, so inspect the columns on first run and we'll map
    # them into your auction_prices-style schema once confirmed.
    return pd.json_normalize(data)


def get_jao_auction_results(
    corridor: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: str = "Daily",
) -> pd.DataFrame:
    """
    Fetch explicit auction results (capacity + clearing price) for a single
    corridor between start and end, chunking into JAO's 31-day request limit.

    Example:
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp.now(tz="UTC")
        df = get_jao_auction_results("GB-FR", start, end)
    """
    chunks = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS), end)
        chunk = _fetch_auctions_chunk(corridor, chunk_start, chunk_end, horizon)
        if not chunk.empty:
            chunks.append(chunk)
        chunk_start = chunk_end

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def _explode_auction_results(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten getauctions' nested per-day, per-hour structure into one row
    per (corridor, delivery hour), which is what actually needs writing.
    """
    rows = []

    for _, auction in raw.iterrows():
        market_period_start = pd.to_datetime(auction["marketPeriodStart"], utc=True)

        for result in auction.get("results") or []:
            product_hour = result.get("productHour")
            if not product_hour:
                continue

            # productHour looks like "15:00-16:00" -- the start hour is the
            # offset from marketPeriodStart (the UTC instant for local
            # midnight of the delivery day).
            start_hour = int(product_hour.split("-")[0].split(":")[0])
            delivery_start = market_period_start + pd.Timedelta(hours=start_hour)

            rows.append({
                "auction_id": auction["identification"],
                "corridor": auction["corridorCode"],
                "horizon": auction["horizonName"],
                "delivery_start": delivery_start,
                "offered_capacity": result.get("offeredCapacity"),
                "requested_capacity": result.get("requestedCapacity"),
                "allocated_capacity": result.get("allocatedCapacity"),
                "price": result.get("auctionPrice"),
            })

    return pd.DataFrame(rows)


def write_ic_auction_results_to_table(results: pd.DataFrame) -> None:
    """
    Upsert JAO interconnector auction results into the ic_auction_prices
    table, keyed on (corridor, delivery_start) -- each getauctions row
    covers a whole day and packs 24 hourly results, so this explodes them
    into individual hourly rows first, same granularity as auction_prices.
    """
    if results.empty:
        print("No results to write.")
        return

    records = _explode_auction_results(results)

    if records.empty:
        print("No hourly results found after exploding response, skipping write.")
        return

    records["delivery_start"] = records["delivery_start"].apply(lambda ts: ts.isoformat())
    for col in ("offered_capacity", "requested_capacity", "allocated_capacity", "price"):
        records[col] = records[col].astype(float)

    db = SupabaseDB()
    db.write(
        "ic_auction_prices",
        records.to_dict(orient="records"),
        upsert=True,
        on_conflict="corridor,delivery_start",
    )

def get_ic_da_prices(start, end):
    for name, (forward, reverse) in CORRIDORS.items():
        for corridor in (forward, reverse):
            print(f"\nFetching {name} ({corridor})...")
            df = get_jao_auction_results(corridor, start, end)
            write_ic_auction_results_to_table(df)

if __name__ == "__main__":
    # Step 1: confirm the real corridor codes before trusting CORRIDORS above.
    print("Available corridors:")
    print(get_corridors())

    print("\nAvailable horizons:")
    print(get_horizons())

    # Step 2: once corridor codes are confirmed, fetch a short recent window
    # per interconnector to inspect the actual response shape.
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    end = pd.Timestamp.now(tz="UTC")

