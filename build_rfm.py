
"""Build the RFM table from cleaned Instamart data."""

from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"

ORDERS_FILE = DATA_DIR / "orders.csv"
CUSTOMERS_FILE = DATA_DIR / "customers.csv"
OUTPUT_FILE = OUTPUT_DIR / "rfm_table.csv"

# Snapshot moment
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    orders = pd.read_csv(ORDERS_FILE)
    customers = pd.read_csv(CUSTOMERS_FILE)

    # Convert dates
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Use completed orders only
    completed = orders.loc[orders["status"] == "completed"].copy()

    # Calculate RFM values for each customer
    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    # Recency = whole days since the latest completed order
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days

    # Remove the intermediate date column
    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ]

    # Keep a consistent column order and data types
    rfm["frequency"] = rfm["frequency"].astype(int)
    rfm["monetary"] = rfm["monetary"].round(2)

    # Save output
    rfm.to_csv(OUTPUT_FILE, index=False)

    # Print reference-checkpoint information
    print("RFM table shape:", rfm.shape)
    print("Unique customers:", rfm["customer_id"].nunique())
    print("Recency median:", rfm["recency"].median())
    print("Recency minimum:", rfm["recency"].min())
    print("Recency maximum:", rfm["recency"].max())
    print("Frequency mean:", round(rfm["frequency"].mean(), 2))
    print("Frequency maximum:", rfm["frequency"].max())
    print("Monetary mean (INR):", round(rfm["monetary"].mean(), 2))
    print("Monetary maximum (INR):", round(rfm["monetary"].max(), 2))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
