
"""Oversample the training data and compare lapse-model performance."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.utils import resample

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = OUTPUT_DIR / "lapse_dataset.csv"
MODEL_FILE = OUTPUT_DIR / "lapse_model.joblib"
RESULTS_FILE = OUTPUT_DIR / "oversampling_comparison.csv"

TARGET = "lapsed"

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


def report(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_lapsed": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_lapsed": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_lapsed": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    print(f"\n{name}")
    print("Accuracy:", round(result["accuracy"], 4))
    print("Precision:", round(result["precision_lapsed"], 4))
    print("Recall:", round(result["recall_lapsed"], 4))
    print("F1:", round(result["f1_lapsed"], 4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    return result


def main():
    data = pd.read_csv(DATA_FILE)
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    # Recreate the exact original split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Load the selected, already fitted Random Forest pipeline.
    original_model = joblib.load(MODEL_FILE)

    print("Original training counts:")
    print(y_train.value_counts().sort_index().to_dict())
    print("Test counts (unchanged):")
    print(y_test.value_counts().sort_index().to_dict())

    original_result = report(
        "Original Random Forest",
        original_model,
        X_test,
        y_test,
    )

    # Resample training rows only, with replacement.
    train_df = X_train.copy()
    train_df[TARGET] = y_train.to_numpy()

    majority = train_df[train_df[TARGET] == 0]
    minority = train_df[train_df[TARGET] == 1]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )

    balanced_train = pd.concat(
        [majority, minority_upsampled],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    X_balanced = balanced_train[FEATURES]
    y_balanced = balanced_train[TARGET].astype(int)

    print("\nBalanced training counts:")
    print(y_balanced.value_counts().sort_index().to_dict())

    # Clone the fitted pipeline's configuration and refit on balanced data.
    balanced_model = clone(original_model)
    balanced_model.fit(X_balanced, y_balanced)

    balanced_result = report(
        "Oversampled Random Forest",
        balanced_model,
        X_test,
        y_test,
    )

    comparison = pd.DataFrame([original_result, balanced_result])
    comparison.to_csv(RESULTS_FILE, index=False)

    # Save this as a separate model so the original is preserved.
    balanced_model_path = OUTPUT_DIR / "lapse_model_oversampled.joblib"
    joblib.dump(balanced_model, balanced_model_path)

    print("\nSaved comparison:", RESULTS_FILE)
    print("Saved oversampled model:", balanced_model_path)


if __name__ == "__main__":
    main()
