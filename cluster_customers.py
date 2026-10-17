
"""Evaluate K-means clustering for customer RFM data."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "rfm_table.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_FILE = OUTPUT_DIR / "kmeans_evaluation.csv"
SCALED_FILE = OUTPUT_DIR / "rfm_scaled.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the RFM table
    rfm = pd.read_csv(INPUT_FILE)

    features = ["recency", "frequency", "monetary"]
    X = rfm[features]

    # Standardize features before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save standardized values for reuse in later steps
    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_df.insert(0, "customer_id", rfm["customer_id"])
    scaled_df.to_csv(SCALED_FILE, index=False)

    # Evaluate K = 2 through K = 8
    results = []

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_
        silhouette = silhouette_score(X_scaled, labels)

        results.append(
            {
                "k": k,
                "inertia": round(inertia, 1),
                "silhouette": round(silhouette, 3),
            }
        )

        print(
            f"K={k}: inertia={inertia:.1f}, "
            f"silhouette={silhouette:.3f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)

    print("\nK-means evaluation:")
    print(results_df.to_string(index=False))
    print("\nSaved evaluation to:", RESULTS_FILE)
    print("Saved standardized RFM data to:", SCALED_FILE)


if __name__ == "__main__":
    main()
