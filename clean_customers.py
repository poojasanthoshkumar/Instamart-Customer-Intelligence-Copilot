
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(RAW_DIR / "customers.csv")

    # Convert blank strings to missing values for cleaning.
    customers["city"] = customers["city"].astype("string").str.strip().str.title()
    customers["gender"] = customers["gender"].astype("string").str.strip().str.lower()

    # Standardise gender spellings.
    gender_map = {
        "m": "male",
        "male": "male",
        "man": "male",
        "men": "male",
        "f": "female",
        "female": "female",
        "woman": "female",
        "women": "female",
    }
    customers["gender"] = customers["gender"].map(gender_map).fillna("unknown")

    # Convert invalid ages to missing values.
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    invalid_age = (customers["age"] < 10) | (customers["age"] > 100)
    invalid_age_count = int(invalid_age.sum())
    customers.loc[invalid_age, "age"] = pd.NA

    missing_age_before = int(customers["age"].isna().sum())

    # Fill missing ages using the median of the customer's city.
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians)

    # Convert signup date to datetime.
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    # Ensure age values are integers after median imputation.
    customers["age"] = customers["age"].round().astype("Int64")

    # Save dates in a consistent text format in the CSV.
    output = customers.copy()
    output["signup_date"] = output["signup_date"].dt.strftime("%Y-%m-%d")
    output.to_csv(CLEAN_DIR / "customers.csv", index=False)

    print("CUSTOMER CLEANING SUMMARY")
    print("=" * 40)
    print("City counts:")
    print(customers["city"].value_counts().to_string())
    print("\nGender counts:")
    print(customers["gender"].value_counts().to_string())
    print(f"\nInvalid ages set to missing: {invalid_age_count}")
    print(f"Missing ages before imputation: {missing_age_before}")
    print(f"Missing ages after imputation: {customers['age'].isna().sum()}")
    print(f"Minimum age: {customers['age'].min()}")
    print(f"Maximum age: {customers['age'].max()}")
    print(f"Missing signup dates: {customers['signup_date'].isna().sum()}")
    print(f"\nSaved: {CLEAN_DIR / 'customers.csv'}")


if __name__ == "__main__":
    main()
