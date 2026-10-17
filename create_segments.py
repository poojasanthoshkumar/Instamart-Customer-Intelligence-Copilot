
"""Create and profile four customer segments using RFM data."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

RFM_FILE = OUTPUT_DIR / "rfm_table.csv"
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"
PROFILE_FILE = OUTPUT_DIR / "segment_profiles.csv"
PCA_FILE = OUTPUT_DIR / "pca_coordinates.csv"

FEATURES = ["recency", "frequency", "monetary"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the RFM dataset
    rfm = pd.read_csv(RFM_FILE)
    X = rfm[FEATURES]

    # 2. Scale RFM values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Fit the selected K-means model
    model = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42,
    )
    rfm["segment_id"] = model.fit_predict(X_scaled)

    # 4. Calculate descriptive statistics for each segment
    profile = (
        rfm.groupby("segment_id")
        .agg(
            customer_count=("customer_id", "nunique"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
            median_recency=("recency", "median"),
            median_frequency=("frequency", "median"),
            median_monetary=("monetary", "median"),
        )
        .reset_index()
    )

    # 5. Identify the highest-spending segment as Premium
    premium_id = int(
        profile.loc[
            profile["mean_monetary"].idxmax(), "segment_id"
        ]
    )

    median_recency = rfm["recency"].median()
    median_frequency = rfm["frequency"].median()

    # 6. Assign business names and recommended actions
    def classify_segment(row):
        sid = int(row["segment_id"])

        if sid == premium_id:
            return (
                "Premium",
                "Offer personalised rewards, loyalty benefits, "
                "and priority customer support."
            )

        # Customers with substantially higher recency are
        # classified as dormant, indicating a longer absence.
        if row["mean_recency"] >= 2 * median_recency:
            return (
                "Dormant Customers",
                "Run targeted win-back campaigns and relevant "
                "offers to encourage repeat purchases."
            )

        if row["mean_recency"] >= median_recency:
            return (
                "At-Risk Customers",
                "Send personalised re-engagement offers and "
                "monitor declining purchase activity."
            )

        if row["mean_frequency"] >= median_frequency:
            return (
                "Loyal Customers",
                "Maintain engagement through loyalty rewards, "
                "cross-selling, and repeat-purchase incentives."
            )

        return (
            "Occasional Customers",
            "Encourage more frequent purchases using relevant "
            "recommendations and limited-time offers."
        )

    classifications = profile.apply(classify_segment, axis=1)

    profile["segment_name"] = classifications.apply(lambda x: x[0])
    profile["recommended_action"] = classifications.apply(
        lambda x: x[1]
    )

    # 7. Ensure every segment has a unique name
    used_names = set()

    for idx in profile.index:
        name = profile.loc[idx, "segment_name"]

        if name in used_names:
            sid = int(profile.loc[idx, "segment_id"])
            profile.loc[idx, "segment_name"] = (
                f"{name} (Group {sid})"
            )

        used_names.add(profile.loc[idx, "segment_name"])

    # 8. Attach segment labels to individual customers
    name_map = dict(
        zip(profile["segment_id"], profile["segment_name"])
    )

    rfm["segment_name"] = rfm["segment_id"].map(name_map)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

    # 9. Save the required customer-level segment output
    required_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]

    rfm[required_columns].to_csv(SEGMENTS_FILE, index=False)

    # 10. Save segment profiles and recommended actions
    profile.to_csv(PROFILE_FILE, index=False)

    # 11. Calculate PCA coordinates for visualisation
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "customer_id": rfm["customer_id"],
        "segment_id": rfm["segment_id"],
        "PC1": coordinates[:, 0],
        "PC2": coordinates[:, 1],
    })

    pca_df.to_csv(PCA_FILE, index=False)

    # 12. Display results and validation checks
    print("\nSegment profiles:")
    print(
        profile[
            [
                "segment_id",
                "customer_count",
                "mean_recency",
                "mean_frequency",
                "mean_monetary",
                "segment_name",
                "recommended_action",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nTotal customers:", len(rfm))
    print("Number of segments:", rfm["segment_id"].nunique())
    print("Premium segment ID:", premium_id)

    print(
        "PCA explained variance ratio:",
        pca.explained_variance_ratio_.round(3),
    )
    print(
        "Total PCA variance explained:",
        round(pca.explained_variance_ratio_.sum(), 3),
    )

    print("\nSaved:", SEGMENTS_FILE)
    print("Saved:", PROFILE_FILE)
    print("Saved:", PCA_FILE)


if __name__ == "__main__":
    main()
