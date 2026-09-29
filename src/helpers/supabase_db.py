from pathlib import Path
import os
from typing import Any

from dotenv import load_dotenv
from supabase import create_client, Client
from helpers.check_env import check_env

check_env()



class SupabaseDB:
    """Thin wrapper around the Supabase client for reading and writing tables."""

    def __init__(
        self,
        env_file: Path | None = None,
        url_var: str = "eu_power_sbdb_URL",
        key_var: str = "eu_power_sbdb_KEY",
    ) -> None:
        self.env_file = env_file or Path.home() / ".env"
        self._load_env()

        self.url = os.getenv(url_var)
        self.key = os.getenv(key_var)

        if not self.url or not self.key:
            raise EnvironmentError(
                f"Missing {url_var} or {key_var}.\n"
                f"Loaded .env from: {self.env_file}\n"
                f"URL found: {bool(self.url)}\n"
                f"KEY found: {bool(self.key)}"
            )

        self.client: Client = create_client(self.url, self.key)

    def _load_env(self) -> None:
        if not self.env_file.exists():
            raise FileNotFoundError(f"Expected .env file at: {self.env_file}")
        load_dotenv(dotenv_path=self.env_file, override=True)

    def write(
        self,
        table: str,
        records: list[dict[str, Any]] | dict[str, Any],
        upsert: bool = False,
        on_conflict: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Insert (or upsert) one or more records into a table.

        Example:
            db.write(
                "auction_prices",
                {"delivery_start": "2026-07-21T00:00:00Z", "region": "DE", "auction": "day-ahead", "price": 45.2},
                upsert=True,
                on_conflict="delivery_start,region",
            )
        """
        if isinstance(records, dict):
            records = [records]

        query = self.client.table(table)
        if upsert:
            response = query.upsert(records, on_conflict=on_conflict).execute()
        else:
            response = query.insert(records).execute()

        return response.data

    def read(
        self,
        table: str,
        columns: str = "*",
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        desc: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Read rows from a table, optionally filtered by equality on given columns.

        Example:
            db.read("auction_prices", filters={"region": "DE"}, order_by="delivery_start", limit=100)
        """
        query = self.client.table(table).select(columns)

        if filters:
            for column, value in filters.items():
                query = query.eq(column, value)

        if order_by:
            query = query.order(order_by, desc=desc)

        if limit:
            query = query.limit(limit)

        response = query.execute()
        return response.data


if __name__ == "__main__":

    db = SupabaseDB()

    # Example read
    rows = db.read("auction_prices", filters={"region": "DE"}, order_by="delivery_start", limit=10)
    print(rows)