
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
DB_PATH = BASE_DIR / "data" / "instamart.db"


def main():
    required_files = [
        CLEAN_DIR / "customers.csv",
        CLEAN_DIR / "products.csv",
        CLEAN_DIR / "orders.csv",
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing cleaned file: {file_path}")

    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="raise"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Remove the previous database before creating a fresh one.
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as connection:
        customers.to_sql("customers", connection, index=False)
        products.to_sql("products", connection, index=False)
        orders.to_sql("orders", connection, index=False)

        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """,
            connection,
        )["name"].tolist()

    expected_tables = ["customers", "orders", "products"]

    if tables != expected_tables:
        raise RuntimeError(f"Unexpected database tables: {tables}")

    print("DATABASE BUILD SUMMARY")
    print("=" * 40)
    print(f"Customers loaded: {len(customers)}")
    print(f"Products loaded: {len(products)}")
    print(f"Orders loaded: {len(orders)}")
    print(f"Tables in database: {', '.join(tables)}")
    print(f"Database saved: {DB_PATH}")


if __name__ == "__main__":
    main()
