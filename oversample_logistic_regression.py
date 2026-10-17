
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.utils import resample


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

DATA_PATH = OUTPUTS / "lapse_dataset.csv"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age",
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "favourite_category",
]


# --------------------------------------------------
# 2. Load the lapse dataset
# --------------------------------------------------

data = pd.read_csv(DATA_PATH)

X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = data["lapsed"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training rows before oversampling:", len(X_train))
print("Test rows:", len(X_test))
print("Original training class counts:")
print(y_train.value_counts().sort_index())


# --------------------------------------------------
# 3. Build preprocessing and Logistic Regression
# --------------------------------------------------

def make_pipeline(C=1.0):
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        (
            "onehot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES,
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(max_iter=1000, C=C),
        ),
    ])


# --------------------------------------------------
# 4. Tune Logistic Regression on original training data
# --------------------------------------------------

grid = GridSearchCV(
    estimator=make_pipeline(),
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

original_model = grid.best_estimator_

print("\nBest Logistic Regression parameters:", grid.best_params_)
print("Best cross-validation F1:", round(grid.best_score_, 4))


# --------------------------------------------------
# 5. Oversample TRAINING DATA ONLY
# --------------------------------------------------

training_data = X_train.copy()
training_data["lapsed"] = y_train.to_numpy()

majority_count = training_data["lapsed"].value_counts().max()

balanced_parts = []

for class_value in sorted(training_data["lapsed"].unique()):
    class_rows = training_data[
        training_data["lapsed"] == class_value
    ]

    balanced_class = resample(
        class_rows,
        replace=(len(class_rows) < majority_count),
        n_samples=majority_count,
        random_state=RANDOM_STATE,
    )

    balanced_parts.append(balanced_class)

balanced_train = pd.concat(
    balanced_parts,
    ignore_index=True,
)

# Shuffle the balanced training rows.
balanced_train = balanced_train.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

X_balanced = balanced_train[
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
]
y_balanced = balanced_train["lapsed"].astype(int)

print("\nTraining class counts after oversampling:")
print(y_balanced.value_counts().sort_index())


# --------------------------------------------------
# 6. Refit Logistic Regression on balanced training data
# --------------------------------------------------

best_C = grid.best_params_["model__C"]

oversampled_model = make_pipeline(C=best_C)
oversampled_model.fit(X_balanced, y_balanced)


# --------------------------------------------------
# 7. Evaluate both models on the SAME untouched test set
# --------------------------------------------------

def evaluate_model(model, X_eval, y_eval, model_name):
    predictions = model.predict(X_eval)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score(
            y_eval, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_eval, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_eval, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_eval, predictions
        ).tolist(),
    }


original_results = evaluate_model(
    original_model,
    X_test,
    y_test,
    "Logistic Regression - Original Training Data",
)

oversampled_results = evaluate_model(
    oversampled_model,
    X_test,
    y_test,
    "Logistic Regression - Oversampled Training Data",
)

comparison = pd.DataFrame([
    {
        key: value
        for key, value in result.items()
        if key != "confusion_matrix"
    }
    for result in [original_results, oversampled_results]
])

comparison_path = OUTPUTS / "logistic_oversampling_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))

print("\nOriginal model confusion matrix:")
print(np.array(original_results["confusion_matrix"]))

print("\nOversampled model confusion matrix:")
print(np.array(oversampled_results["confusion_matrix"]))


# --------------------------------------------------
# 8. Save the oversampled model separately
# --------------------------------------------------

model_path = OUTPUTS / "lapse_logistic_oversampled.joblib"
joblib.dump(oversampled_model, model_path)

# Save a detailed comparison, including confusion matrices.
details_path = OUTPUTS / "logistic_oversampling_details.json"

with open(details_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "best_parameters": grid.best_params_,
            "best_cv_f1": float(grid.best_score_),
            "original_model": original_results,
            "oversampled_model": oversampled_results,
            "training_counts_before": {
                str(k): int(v)
                for k, v in y_train.value_counts().items()
            },
            "training_counts_after": {
                str(k): int(v)
                for k, v in y_balanced.value_counts().items()
            },
            "test_set_unchanged": True,
        },
        file,
        indent=2,
    )

# Confirm that the saved model reloads correctly.
reloaded_model = joblib.load(model_path)

predictions_before = oversampled_model.predict(X_test)
predictions_after = reloaded_model.predict(X_test)

print(
    "\nSaved-model prediction equality:",
    np.array_equal(predictions_before, predictions_after),
)

print("\nCreated files:")
print(comparison_path)
print(details_path)
print(model_path)
