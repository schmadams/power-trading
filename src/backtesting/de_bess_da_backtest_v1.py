import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

from helpers.check_env import check_env
from helpers.supabase_db import SupabaseDB

check_env()

# Supabase/PostgREST caps each request at 1000 rows by default, so wide
# date ranges need paginating with .range() rather than a single .execute().
PAGE_SIZE = 1000

# Assumption: region code for Germany in the auction_prices table is "DE_LU"
# (post-2018 German/Luxembourg bidding zone). Change if your table uses "DE".
DE_REGION = "DE_LU"

# BESS specs
POWER_MW = 1.0
ENERGY_MWH = 2.0
ROUND_TRIP_EFFICIENCY = 0.8

# Split round-trip efficiency evenly across charge and discharge legs
CHARGE_EFFICIENCY = np.sqrt(ROUND_TRIP_EFFICIENCY)
DISCHARGE_EFFICIENCY = np.sqrt(ROUND_TRIP_EFFICIENCY)


def read_de_da_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Read DE day-ahead auction prices from the auction_prices table for the
    given delivery_start date range, paginating as needed.
    """
    db = SupabaseDB()
    rows = []
    offset = 0

    while True:
        response = (
            db.client.table("auction_prices")
            .select("delivery_start, region, auction, price")
            .eq("region", DE_REGION)
            .eq("auction", "day-ahead")
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


def optimise_day(prices: np.ndarray, dt_hours: float) -> dict:
    """
    Solve the perfect-foresight dispatch LP for a single day, treated as
    independent (SoC starts and ends at 0). Decision variables are hourly
    charge and discharge power, [c_1..c_T, d_1..d_T].
    """
    n_periods = len(prices)

    # Objective: minimise (charge cost - discharge revenue), i.e.
    # minimise price*c - price*d, since linprog minimises by default.
    cost_vector = np.concatenate([prices, -prices])

    # Power limits: 0 <= c_t <= POWER_MW, 0 <= d_t <= POWER_MW
    bounds = [(0, POWER_MW)] * n_periods + [(0, POWER_MW)] * n_periods

    # SoC at end of period t = dt * cumsum(charge_eff * c - d / discharge_eff)
    # Need 0 <= SoC_t <= ENERGY_MWH for every t, and SoC_T = 0.
    charge_contribution = dt_hours * CHARGE_EFFICIENCY
    discharge_contribution = dt_hours / DISCHARGE_EFFICIENCY

    lower_tri = np.tril(np.ones((n_periods, n_periods)))
    soc_from_charge = lower_tri * charge_contribution
    soc_from_discharge = lower_tri * discharge_contribution

    # SoC_t <= ENERGY_MWH  ->  soc_from_charge @ c - soc_from_discharge @ d <= ENERGY_MWH
    A_ub_upper = np.hstack([soc_from_charge, -soc_from_discharge])
    b_ub_upper = np.full(n_periods, ENERGY_MWH)

    # SoC_t >= 0  ->  -soc_from_charge @ c + soc_from_discharge @ d <= 0
    A_ub_lower = np.hstack([-soc_from_charge, soc_from_discharge])
    b_ub_lower = np.zeros(n_periods)

    A_ub = np.vstack([A_ub_upper, A_ub_lower])
    b_ub = np.concatenate([b_ub_upper, b_ub_lower])

    # SoC_T = 0 (end the day empty)
    A_eq = np.hstack([soc_from_charge[-1:], -soc_from_discharge[-1:]])
    b_eq = np.array([0.0])

    result = linprog(
        cost_vector,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return {"profit": np.nan, "charge": None, "discharge": None}

    charge = result.x[:n_periods]
    discharge = result.x[n_periods:]
    profit = float(-result.fun)

    return {"profit": profit, "charge": charge, "discharge": discharge}


def run_backtest(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the daily independent-SoC optimisation across all days present in
    prices_df and return a per-day profit summary.
    """
    prices_df = prices_df.sort_values("delivery_start").copy()
    prices_df["date"] = prices_df["delivery_start"].dt.date

    daily_results = []

    for date, day_df in prices_df.groupby("date"):
        day_df = day_df.sort_values("delivery_start")
        prices = day_df["price"].to_numpy()

        # Infer resolution in hours from consecutive timestamps
        deltas = day_df["delivery_start"].diff().dropna()
        dt_hours = deltas.dt.total_seconds().median() / 3600 if not deltas.empty else 1.0

        result = optimise_day(prices, dt_hours)
        daily_results.append({"date": date, "n_periods": len(prices), "profit_eur": result["profit"]})

    return pd.DataFrame(daily_results)


def plot_daily_pnl(daily_profit: pd.DataFrame) -> None:
    """
    Plot daily and cumulative PnL from the backtest results.
    """
    daily_profit = daily_profit.sort_values("date").copy()
    daily_profit["cumulative_profit_eur"] = daily_profit["profit_eur"].cumsum()

    fig = px.bar(
        daily_profit,
        x="date",
        y="profit_eur",
        title="Daily PnL",
        labels={"date": "Date", "profit_eur": "Daily profit (EUR)"},
    )
    fig.add_scatter(
        x=daily_profit["date"],
        y=daily_profit["cumulative_profit_eur"],
        name="Cumulative profit",
        yaxis="y2",
        mode="lines",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis2=dict(title="Cumulative profit (EUR)", overlaying="y", side="right"),
        showlegend=False,
    )
    fig.show()


if __name__ == "__main__":
    start = pd.Timestamp("2022-01-01", tz="Europe/Brussels")
    end = pd.Timestamp.now(tz="Europe/Brussels")

    prices_df = read_de_da_prices(start, end)
    daily_profit = run_backtest(prices_df)

    print(daily_profit)
    print(f"\nTotal profit: EUR {daily_profit['profit_eur'].sum():,.0f}")
    print(f"Average daily profit: EUR {daily_profit['profit_eur'].mean():,.0f}")

    plot_daily_pnl(daily_profit)