
"""Build a leakage-aware customer lapse dataset."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"

CUSTOMERS_FILE = DATA_DIR / "customers.csv"
ORDERS_FILE = DATA_DIR / "orders.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
OUTPUT_FILE = OUTPUT_DIR / "lapse_dataset.csv"

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(CUSTOMERS_FILE)
    orders = pd.read_csv(ORDERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    # Historical data: features must use only orders before the cutoff.
    history = orders.loc[orders["order_ts"] < CUTOFF].copy()

    # The cohort includes customers with at least one completed
    # order before the cutoff.
    completed_history = history.loc[
        history["status"] == "completed"
    ].copy()

    cohort_ids = completed_history["customer_id"].unique()

    # Build RFM-style historical features.
    features = (
        completed_history.groupby("customer_id")
        .agg(
            last_completed_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
        )
        .reset_index()
    )

    features["recency"] = (
        CUTOFF - features["last_completed_order"]
    ).dt.days

    features = features.drop(columns=["last_completed_order"])

    # Average discount from all historical orders, including cancellations.
    discount = (
        history.groupby("customer_id")["discount_pct"]
        .mean()
        .rename("avg_discount")
    )

    # Cancellation rate among all historical orders.
    history["is_cancelled"] = (
        history["status"] == "cancelled"
    ).astype(int)

    cancellation_rate = (
        history.groupby("customer_id")["is_cancelled"]
        .mean()
        .rename("cancellation_rate")
    )

    # Most frequently ordered product category based on completed
    # historical orders.
    category_orders = completed_history.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
    )

    category_counts = (
        category_orders.groupby(["customer_id", "category"])
        .size()
        .reset_index(name="category_order_count")
        .sort_values(
            ["customer_id", "category_order_count", "category"],
            ascending=[True, False, True],
        )
    )

    favourite_category = (
        category_counts.drop_duplicates("customer_id")
        .set_index("customer_id")["category"]
        .rename("favourite_category")
    )

    # Combine historical features.
    features = features.set_index("customer_id")
    features = features.join(discount)
    features = features.join(cancellation_rate)
    features = features.join(favourite_category)
    features = features.reset_index()

    # Add customer information and tenure as of the cutoff.
    customer_features = customers[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ].copy()

    customer_features["tenure_days"] = (
        CUTOFF - customer_features["signup_date"]
    ).dt.days

    customer_features = customer_features.drop(columns=["signup_date"])

    dataset = customer_features.merge(
        features,
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )

    # Label using September activity only.
    september_completed = orders.loc[
        (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < LABEL_END)
        & (orders["status"] == "completed"),
        "customer_id",
    ].unique()

    dataset["lapsed"] = (
        ~dataset["customer_id"].isin(september_completed)
    ).astype(int)

    # Fill missing historical delivery averages using a fixed
    # historical-data median, not September information.
    historical_delivery_median = completed_history[
        "delivery_minutes"
    ].median()

    dataset["avg_delivery_minutes"] = dataset[
        "avg_delivery_minutes"
    ].fillna(historical_delivery_median)

    # Fail early if any other feature is missing.
    missing = dataset.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            "Missing values found in lapse dataset:\n"
            + missing.to_string()
        )

    # Save with the label at the end.
    columns = [
        "customer_id",
        "city",
        "age",
        "gender",
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery_minutes",
        "favourite_category",
        "tenure_days",
        "lapsed",
    ]

    dataset = dataset[columns]
    dataset.to_csv(OUTPUT_FILE, index=False)

    print("Lapse dataset shape:", dataset.shape)
    print("Unique customers:", dataset["customer_id"].nunique())
    print("\nLapse label counts:")
    print(dataset["lapsed"].value_counts().sort_index())
    print("\nLapsed percentage:", round(dataset["lapsed"].mean() * 100, 2))
    print("\nMissing values:")
    print(dataset.isna().sum().to_string())
    print("\nFeature columns:")
    print(dataset.columns.tolist())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
