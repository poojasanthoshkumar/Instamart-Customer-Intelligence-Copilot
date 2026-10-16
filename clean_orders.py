
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw orders and product catalogue.
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")

    initial_rows = len(orders)

    # 1. Parse order timestamps.
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    # 2. Remove exact duplicate records.
    exact_duplicates = int(orders.duplicated().sum())
    orders = orders.drop_duplicates().copy()

    # 3. Keep the latest record for each order_id.
    orders = orders.sort_values("order_ts")
    orders = orders.drop_duplicates(
        subset="order_id", keep="last"
    ).copy()

    # Match each order to its product details.
    products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

    orders = orders.merge(
        products[["product_id", "category", "price_inr"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # 4. Convert amounts to numeric and calculate missing amounts.
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    missing_amount_before = int(orders["order_amount"].isna().sum())

    orders["quantity"] = pd.to_numeric(
        orders["quantity"], errors="coerce"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )

    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    orders["order_amount"] = orders["order_amount"].fillna(
        calculated_amount
    ).round(2)

    # 5. Replace negative delivery times with missing values.
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    negative_delivery = orders["delivery_minutes"] < 0
    negative_delivery_count = int(negative_delivery.sum())
    orders.loc[negative_delivery, "delivery_minutes"] = pd.NA

    # 6. Fill missing delivery times using the median for the
    # same product category among completed orders only.
    completed = orders["status"].astype(str).str.strip().str.lower().eq(
        "completed"
    )

    orders.loc[completed, "delivery_minutes"] = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform(lambda values: values.fillna(values.median()))
    )

    # Cancelled orders must retain missing delivery times.
    cancelled = orders["status"].astype(str).str.strip().str.lower().eq(
        "cancelled"
    )
    orders.loc[cancelled, "delivery_minutes"] = pd.NA

    # Store timestamps in the required text format.
    orders["order_ts"] = orders["order_ts"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Save only the required order columns; category and price
    # are used for cleaning, not stored in the orders table.
    output_columns = [
        "order_id",
        "customer_id",
        "product_id",
        "order_ts",
        "quantity",
        "order_amount",
        "discount_pct",
        "delivery_minutes",
        "status",
    ]

    clean_orders = orders[output_columns].copy()
    clean_orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

    # Report cleaning results.
    print("ORDER CLEANING SUMMARY")
    print("=" * 40)
    print(f"Initial raw rows: {initial_rows}")
    print(f"Exact duplicate rows removed: {exact_duplicates}")
    print(f"Final rows: {len(clean_orders)}")
    print(f"Distinct order IDs: {clean_orders['order_id'].nunique()}")
    print("\nOrder status counts:")
    print(clean_orders["status"].value_counts(dropna=False).to_string())
    print(f"\nMissing amounts before filling: {missing_amount_before}")
    print(f"Missing amounts after filling: {clean_orders['order_amount'].isna().sum()}")
    print(f"Negative delivery times set to missing: {negative_delivery_count}")

    final_completed = clean_orders["status"].astype(str).str.strip().str.lower().eq(
        "completed"
    )
    final_cancelled = clean_orders["status"].astype(str).str.strip().str.lower().eq(
        "cancelled"
    )

    print(
        "Missing delivery times among completed orders: "
        f"{clean_orders.loc[final_completed, 'delivery_minutes'].isna().sum()}"
    )
    print(
        "Missing delivery times among cancelled orders: "
        f"{clean_orders.loc[final_cancelled, 'delivery_minutes'].isna().sum()}"
    )
    print(
        "Completed order amount total: INR "
        f"{clean_orders.loc[final_completed, 'order_amount'].sum():,.2f}"
    )
    print(f"\nSaved: {CLEAN_DIR / 'orders.csv'}")


if __name__ == "__main__":
    main()
