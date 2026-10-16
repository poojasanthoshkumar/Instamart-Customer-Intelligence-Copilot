
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

    completed = orders[
        orders["status"].astype(str).str.lower().eq("completed")
    ].copy()

    completed["order_amount"] = pd.to_numeric(
        completed["order_amount"], errors="coerce"
    )
    completed["discount_pct"] = pd.to_numeric(
        completed["discount_pct"], errors="coerce"
    )
    completed["quantity"] = pd.to_numeric(
        completed["quantity"], errors="coerce"
    )
    completed["delivery_minutes"] = pd.to_numeric(
        completed["delivery_minutes"], errors="coerce"
    )

    # Join data for revenue analysis.
    detailed = (
        completed
        .merge(customers[["customer_id", "city"]], on="customer_id")
        .merge(products[["product_id", "category"]], on="product_id")
    )

    # A. Rebuild SQL Q2 using pandas pivot_table.
    pivot = pd.pivot_table(
        detailed,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All",
    )

    print("=" * 60)
    print("A. PANDAS PIVOT TABLE: REVENUE BY CITY AND CATEGORY")
    print("=" * 60)
    print(pivot.round(2).to_string())

    pandas_total = round(float(pivot.loc["All", "All"]), 2)
    print(f"\nPandas grand total: INR {pandas_total:,.2f}")

    with sqlite3.connect(DB_PATH) as connection:
        sql_total = pd.read_sql_query(
            """
            SELECT SUM(order_amount) AS total
            FROM orders
            WHERE status = 'completed'
            """,
            connection,
        )["total"].iloc[0]

    print(f"SQL grand total: INR {sql_total:,.2f}")
    assert pandas_total == round(float(sql_total), 2)
    assert pandas_total == 2998668.80

    # Grouped city statistics using named aggregation.
    city_stats = (
        detailed.groupby("city")
        .agg(
            total_revenue=("order_amount", "sum"),
            mean_order_amount=("order_amount", "mean"),
            completed_order_count=("order_amount", "count"),
        )
        .sort_index()
    )

    print("\nCity-wise completed-order statistics:")
    print(city_stats.round(2).to_string())

    # B. Descriptive statistics and IQR outlier analysis.
    amount = completed["order_amount"].dropna()
    q1 = amount.quantile(0.25)
    median = amount.median()
    q3 = amount.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    lower_outliers = int((amount < lower_fence).sum())
    upper_outliers = int((amount > upper_fence).sum())

    print("\n" + "=" * 60)
    print("B. COMPLETED ORDER AMOUNT STATISTICS")
    print("=" * 60)
    print(f"Count: {len(amount)}")
    print(f"Mean: INR {amount.mean():.2f}")
    print(f"Median: INR {median:.2f}")
    print(f"Skewness: {amount.skew():.4f}")
    print(f"Minimum: INR {amount.min():.2f}")
    print(f"Q1: INR {q1:.2f}")
    print(f"Q3: INR {q3:.2f}")
    print(f"Maximum: INR {amount.max():.2f}")
    print(f"IQR: INR {iqr:.2f}")
    print(f"Lower 1.5×IQR fence: INR {lower_fence:.2f}")
    print(f"Upper 1.5×IQR fence: INR {upper_fence:.2f}")
    print(f"Lower outliers: {lower_outliers}")
    print(f"Upper outliers: {upper_outliers}")

    # C. Correlation matrix for completed orders.
    correlation_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes",
    ]

    correlation = completed[correlation_columns].corr()

    print("\n" + "=" * 60)
    print("C. CORRELATION MATRIX")
    print("=" * 60)
    print(correlation.round(4).to_string())

    # Identify the strongest absolute off-diagonal correlation.
    abs_corr = correlation.abs()
    for i in range(len(abs_corr.columns)):
        abs_corr.iloc[i, i] = 0

    strongest_pair = abs_corr.stack().idxmax()
    strongest_value = correlation.loc[
        strongest_pair[0], strongest_pair[1]
    ]

    print(
        "\nStrongest absolute correlation: "
        f"{strongest_pair[0]} vs {strongest_pair[1]} "
        f"= {strongest_value:.4f}"
    )

    # D. Compact machine-readable checks.
    assert len(customers) == 1200
    assert len(products) == 40
    assert len(orders) == 12309
    assert len(completed) == 11110
    assert completed["delivery_minutes"].notna().all()

    print("\nALL PANDAS ANALYSIS CHECKS PASSED")


if __name__ == "__main__":
    main()
