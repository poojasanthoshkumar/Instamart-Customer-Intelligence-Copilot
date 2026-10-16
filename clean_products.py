
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(RAW_DIR / "products.csv")

    # Standardise text fields.
    products["product_id"] = products["product_id"].astype("string").str.strip()
    products["product_name"] = products["product_name"].astype("string").str.strip()
    products["category"] = products["category"].astype("string").str.strip().str.title()

    # Ensure prices are numeric and valid.
    products["price_inr"] = pd.to_numeric(
        products["price_inr"], errors="coerce"
    )

    if products["product_id"].isna().any():
        raise ValueError("Products contain missing product IDs.")

    if products["product_id"].duplicated().any():
        raise ValueError("Products contain duplicate product IDs.")

    if products["price_inr"].isna().any() or (products["price_inr"] <= 0).any():
        raise ValueError("Products contain missing or non-positive prices.")

    products.to_csv(CLEAN_DIR / "products.csv", index=False)

    print("PRODUCT CLEANING SUMMARY")
    print("=" * 40)
    print(f"Rows: {len(products)}")
    print(f"Unique product IDs: {products['product_id'].nunique()}")
    print(f"Missing values:\n{products.isna().sum().to_string()}")
    print(f"Categories:\n{products['category'].value_counts().to_string()}")
    print(f"\nSaved: {CLEAN_DIR / 'products.csv'}")


if __name__ == "__main__":
    main()
