import pandas as pd
import plotly.express as px

from helpers.check_env import check_env
from helpers.supabase_db import SupabaseDB

check_env()

# Supabase/PostgREST caps each request at 1000 rows by default, so wide
# date ranges need paginating with .range() rather than a single .execute().
PAGE_SIZE = 1000


def read_da_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Read day-ahead prices from the auction_prices table for the given
    delivery_start date range, across all regions, paginating as needed.
    """
    db = SupabaseDB()
    rows = []
    offset = 0

    while True:
        response = (
            db.client.table("auction_prices")
            .select("delivery_start, region, auction, price")
            .gte("delivery_start", start.isoformat())
            .lte("delivery_start", end.isoformat())
            .order("delivery_start")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )

        page = response.data
        if not page:
            break

        rows.extend(page)
        offset += PAGE_SIZE

        if len(page) < PAGE_SIZE:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["delivery_start"] = pd.to_datetime(df["delivery_start"])

    return df


def plot_da_prices(start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    Read and plot day-ahead prices for the given date range, one line per region.
    """
    df = read_da_prices(start, end)

    if df.empty:
        print("No data found for the given date range.")
        return

    fig = px.line(
        df,
        x="delivery_start",
        y="price",
        color="region",
        title=f"Day-ahead prices — {start.date()} to {end.date()}",
        labels={"delivery_start": "Delivery start", "price": "Price (EUR/MWh)"},
    )
    fig.show()


if __name__ == "__main__":
    start = pd.Timestamp("2026-07-01", tz="Europe/Brussels")
    end = pd.Timestamp.now(tz="Europe/Brussels")

    plot_da_prices(start, end)