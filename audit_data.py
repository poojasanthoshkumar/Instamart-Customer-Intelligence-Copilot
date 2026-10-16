
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"


def load_raw_data():
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")

    # Preserve empty ticket text as an empty string.
    tickets = pd.read_csv(
        RAW_DIR / "tickets.csv",
        keep_default_na=False,
    )

    return customers, orders, products, tickets


def report_table(name, dataframe):
    print(f"\n{'=' * 65}")
    print(f"{name.upper()} — SHAPE: {dataframe.shape}")
    print("=" * 65)

    print("\nDATA TYPES:")
    print(dataframe.dtypes.to_string())

    print("\nMISSING VALUES:")
    missing_count = dataframe.isna().sum()
    missing_percent = (
        dataframe.isna().mean().mul(100).round(2)
    )

    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent,
    })

    print(report.to_string())


def report_reference_audit(customers, orders):
    numeric_amount = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    delivery = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    age = pd.to_numeric(customers["age"], errors="coerce")

    blank_delivery = (
        orders["delivery_minutes"].isna()
        | orders["delivery_minutes"].astype(str).str.strip().eq("")
    )

    blank_age = (
        customers["age"].isna()
        | customers["age"].astype(str).str.strip().eq("")
    )

    blank_gender = (
        customers["gender"].isna()
        | customers["gender"].astype(str).str.strip().eq("")
    )

    print(f"\n{'=' * 65}")
    print("REQUIRED RAW-DATA AUDIT FIGURES")
    print("=" * 65)

    figures = {
        "Raw orders rows": len(orders),
        "Exact duplicate order rows": int(orders.duplicated().sum()),
        "Pending order rows": int(
            orders["status"].astype(str).str.strip().str.lower()
            .eq("pending").sum()
        ),
        "Distinct order IDs": int(orders["order_id"].nunique()),
        "Non-numeric order amounts": int(numeric_amount.isna().sum()),
        "Negative delivery times": int((delivery < 0).sum()),
        "Blank delivery times": int(blank_delivery.sum()),
        "Blank ages": int(blank_age.sum()),
        "Ages outside 10–100": int(
            ((age < 10) | (age > 100)).sum()
        ),
        "Distinct raw city spellings": int(
            customers["city"].nunique()
        ),
        "Distinct non-blank raw gender spellings": int(
            customers.loc[~blank_gender, "gender"].nunique()
        ),
        "Blank genders": int(blank_gender.sum()),
    }

    for label, value in figures.items():
        print(f"{label}: {value}")


def main():
    customers, orders, products, tickets = load_raw_data()

    report_table("Customers", customers)
    report_table("Orders", orders)
    report_table("Products", products)
    report_table("Tickets", tickets)

    report_reference_audit(customers, orders)


if __name__ == "__main__":
    main()
