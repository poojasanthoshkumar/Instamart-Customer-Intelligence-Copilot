
"""Verify saved lapse models and create a Premium retention list."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_FILE = OUTPUT_DIR / "lapse_dataset.csv"
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"
SELECTED_MODEL_FILE = OUTPUT_DIR / "lapse_model.joblib"
OVERSAMPLED_MODEL_FILE = OUTPUT_DIR / "lapse_model_oversampled.joblib"

RETENTION_FILE = OUTPUT_DIR / "retention_list.csv"

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "favourite_category",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    data = pd.read_csv(DATA_FILE)
    segments = pd.read_csv(SEGMENTS_FILE)

    X = data[FEATURES].copy()
    y = data["lapsed"].astype(int)

    # Recreate the same stratified split used during model evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Verify the selected model can be loaded and reproduces predictions.
    selected_model = joblib.load(SELECTED_MODEL_FILE)
    selected_predictions_before = selected_model.predict(X_test)

    reloaded_selected_model = joblib.load(SELECTED_MODEL_FILE)
    selected_predictions_after = reloaded_selected_model.predict(X_test)

    selected_equal = np.array_equal(
        selected_predictions_before,
        selected_predictions_after,
    )

    print("Selected model reload prediction equality:", selected_equal)

    # Verify the oversampled model too.
    oversampled_model = joblib.load(OVERSAMPLED_MODEL_FILE)
    oversampled_predictions_before = oversampled_model.predict(X_test)

    reloaded_oversampled_model = joblib.load(OVERSAMPLED_MODEL_FILE)
    oversampled_predictions_after = reloaded_oversampled_model.predict(X_test)

    oversampled_equal = np.array_equal(
        oversampled_predictions_before,
        oversampled_predictions_after,
    )

    print("Oversampled model reload prediction equality:", oversampled_equal)

    # Identify test-set customer IDs.
    test_ids = set(data.loc[X_test.index, "customer_id"])

    # Retention list uses the oversampled model, which had the higher
    # test F1 and recall in our retention-focused comparison.
    test_data = data[data["customer_id"].isin(test_ids)].copy()

    # Match the segment IDs created earlier.
    test_data = test_data.merge(
        segments[["customer_id", "segment_id", "is_premium"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # CSV boolean values are usually read as True/False, but handle
    # common text representations too.
    premium_mask = test_data["is_premium"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    premium_test = test_data.loc[premium_mask].copy()

    premium_X = premium_test[FEATURES]
    premium_test["lapse_probability"] = (
        oversampled_model.predict_proba(premium_X)[:, 1]
    )

    retention = (
        premium_test[
            ["customer_id", "lapse_probability", "lapsed"]
        ]
        .rename(columns={"lapsed": "actually_lapsed"})
        .sort_values("lapse_probability", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    retention["lapse_probability"] = retention[
        "lapse_probability"
    ].round(4)

    retention.to_csv(RETENTION_FILE, index=False)

    print("\nTop Premium retention candidates:")
    print(retention.to_string(index=False))

    print("\nNumber of customers in retention list:", len(retention))
    print(
        "Actually lapsed among these customers:",
        int(retention["actually_lapsed"].sum()),
    )
    print("\nSaved retention list to:", RETENTION_FILE)

    if not selected_equal or not oversampled_equal:
        raise AssertionError("Model predictions changed after reloading.")


if __name__ == "__main__":
    main()
