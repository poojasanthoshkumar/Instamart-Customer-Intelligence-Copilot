
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

data = pd.read_csv(OUTPUTS / "lapse_dataset.csv")

numeric_features = [
    "age",
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days",
]

X = data[numeric_features]
y = data["lapsed"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
scaler.fit(X_train)

frequency_index = numeric_features.index("frequency")

full_cohort_mean = data["frequency"].mean()
training_scaler_mean = scaler.mean_[frequency_index]

print("Full-cohort mean frequency:", round(full_cohort_mean, 4))
print("Training-set scaler mean frequency:", round(training_scaler_mean, 4))
print("Training customers:", len(X_train))
print("Test customers:", len(X_test))

print(
    "\nExplanation: The means differ because the scaler is fitted only "
    "on the training subset, rather than the full cohort. This prevents "
    "information from the test set influencing preprocessing."
)
