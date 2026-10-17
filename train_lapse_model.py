
"""Train and compare logistic regression and random forest lapse models."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = OUTPUT_DIR / "lapse_dataset.csv"

TARGET = "lapsed"
ID_COL = "customer_id"

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


def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision (lapsed=1):", round(
        precision_score(y_test, predictions, zero_division=0), 4
    ))
    print("Recall (lapsed=1):", round(
        recall_score(y_test, predictions, zero_division=0), 4
    ))
    print("F1 (lapsed=1):", round(
        f1_score(y_test, predictions, zero_division=0), 4
    ))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_FILE)

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data[TARGET].astype(int)

    # Split before fitting preprocessing to avoid leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Cohort size:", len(data))
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Training label counts:")
    print(y_train.value_counts().sort_index().to_dict())
    print("Test label counts:")
    print(y_test.value_counts().sort_index().to_dict())

    # Baseline: always predict not lapsed.
    baseline_predictions = np.zeros(len(y_test), dtype=int)
    print("\nBaseline (always predict not lapsed)")
    print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 4))
    print("Positive-class F1:", round(
        f1_score(y_test, baseline_predictions, zero_division=0), 4
    ))

    # Logistic regression grid search.
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    lr_grid = GridSearchCV(
        lr_pipeline,
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)

    print("\nLogistic Regression best parameters:", lr_grid.best_params_)
    print("Logistic Regression best CV F1:", round(lr_grid.best_score_, 4))
    evaluate_model("Logistic Regression — test set", lr_grid.best_estimator_, X_test, y_test)

    # Random forest grid search.
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    rf_grid = GridSearchCV(
        rf_pipeline,
        param_grid={
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)

    print("\nRandom Forest best parameters:", rf_grid.best_params_)
    print("Random Forest best CV F1:", round(rf_grid.best_score_, 4))
    evaluate_model("Random Forest — test set", rf_grid.best_estimator_, X_test, y_test)

    # Save both search results for later analysis.
    results = {
        "logistic_regression": {
            "best_params": lr_grid.best_params_,
            "best_cv_f1": float(lr_grid.best_score_),
            "test_f1": float(
                f1_score(
                    y_test,
                    lr_grid.best_estimator_.predict(X_test),
                    zero_division=0,
                )
            ),
        },
        "random_forest": {
            "best_params": rf_grid.best_params_,
            "best_cv_f1": float(rf_grid.best_score_),
            "test_f1": float(
                f1_score(
                    y_test,
                    rf_grid.best_estimator_.predict(X_test),
                    zero_division=0,
                )
            ),
        },
    }

    with open(OUTPUT_DIR / "lapse_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Choose the model with the higher cross-validation F1.
    if rf_grid.best_score_ >= lr_grid.best_score_:
        selected_name = "random_forest"
        selected_model = rf_grid.best_estimator_
    else:
        selected_name = "logistic_regression"
        selected_model = lr_grid.best_estimator_

    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)

    print("\nSelected model:", selected_name)
    print("Saved model to:", model_path)
    print("Saved comparison to:", OUTPUT_DIR / "lapse_model_comparison.json")


if __name__ == "__main__":
    main()
