
"""Analyse Random Forest feature importance and Logistic Regression coefficients."""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = OUTPUT_DIR / "lapse_dataset.csv"

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
TARGET = "lapsed"


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def main():
    data = pd.read_csv(DATA_FILE)
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Use the saved oversampled model for Random Forest interpretation.
    rf_model = joblib.load(
        OUTPUT_DIR / "lapse_model_oversampled.joblib"
    )

    preprocessor = rf_model.named_steps["preprocessor"]
    classifier = rf_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importances.to_csv(
        OUTPUT_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    print("\nTop 10 Random Forest features:")
    print(importances.head(10).round(4).to_string(index=False))

    # Fit the best logistic-regression configuration on the original
    # training split for interpretable coefficients.
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

    lr_model = lr_grid.best_estimator_
    lr_names = lr_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    coefficients = lr_model.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": lr_names,
            "coefficient": coefficients,
        }
    ).sort_values("coefficient", ascending=False)

    coef_df.to_csv(
        OUTPUT_DIR / "logistic_regression_coefficients.csv",
        index=False,
    )

    print("\nBest Logistic Regression parameters:", lr_grid.best_params_)
    print("\nTop 5 positive coefficients (associated with higher lapse odds):")
    print(coef_df.head(5).round(4).to_string(index=False))

    print("\nTop 5 negative coefficients (associated with lower lapse odds):")
    print(coef_df.tail(5).sort_values("coefficient").round(4).to_string(index=False))

    print("\nSaved feature importance and coefficient CSV files.")


if __name__ == "__main__":
    main()
