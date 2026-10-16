
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
DB_PATH = BASE_DIR / "data" / "instamart.db"


def main():
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    completed = orders["status"].astype(str).str.lower().eq("completed")
    cancelled = orders["status"].astype(str).str.lower().eq("cancelled")

    checks = {
        "customers_rows": len(customers) == 1200,
        "products_rows": len(products) == 40,
        "orders_rows": len(orders) == 12309,
        "unique_order_ids": orders["order_id"].nunique() == 12309,
        "six_cities": customers["city"].nunique() == 6,
        "customer_ages_complete": customers["age"].notna().all(),
        "customer_ages_in_range": customers["age"].between(10, 100).all(),
        "gender_counts": customers["gender"].value_counts().to_dict()
        == {"female": 610, "male": 560, "unknown": 30},
        "completed_orders": int(completed.sum()) == 11110,
        "cancelled_orders": int(cancelled.sum()) == 1199,
        "no_pending_orders": not orders["status"].astype(str).str.lower().eq("pending").any(),
        "no_missing_amounts": orders["order_amount"].notna().all(),
        "completed_delivery_complete": orders.loc[completed, "delivery_minutes"].notna().all(),
        "cancelled_delivery_missing": orders.loc[cancelled, "delivery_minutes"].isna().all(),
        "completed_revenue": round(orders.loc[completed, "order_amount"].sum(), 2)
        == 2998668.80,
    }

    with sqlite3.connect(DB_PATH) as connection:
        tables = pd.read_sql_query(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """,
            connection,
        )["name"].tolist()

        checks["database_tables"] = tables == [
            "customers", "orders", "products"
        ]

        for table, expected_rows in [
            ("customers", 1200),
            ("products", 40),
            ("orders", 12309),
        ]:
            actual_rows = pd.read_sql_query(
                f"SELECT COUNT(*) AS n FROM {table}", connection
            ).iloc[0]["n"]
            checks[f"database_{table}_rows"] = actual_rows == expected_rows

        stored_timestamps = pd.read_sql_query(
            "SELECT order_ts FROM orders", connection
        )["order_ts"]

        checks["timestamp_format"] = stored_timestamps.str.match(
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
        ).all()

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        print("FAILED CHECKPOINTS:")
        for name in failed:
            print(f"- {name}")
        raise SystemExit(1)

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
