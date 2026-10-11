
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# Paths
DATA_PATH = "DATASET.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset loaded:", df.shape)

# Prepare metadata features
categorical_cols = [
    col for col in ["type", "rating", "listed_in", "country"]
    if col in df.columns
]
numeric_cols = [
    col for col in ["release_year"]
    if col in df.columns
]

features = df[categorical_cols + numeric_cols].copy()

# Handle missing values
for col in categorical_cols:
    features[col] = features[col].fillna("Unknown").astype(str)

for col in numeric_cols:
    features[col] = pd.to_numeric(
        features[col], errors="coerce"
    ).fillna(df[col].median())

# Encode categorical data and scale numeric data
transformers = []

if categorical_cols:
    transformers.append(
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_cols)
    )

if numeric_cols:
    transformers.append(
        ("numeric", StandardScaler(), numeric_cols)
    )

preprocessor = ColumnTransformer(transformers)
X = preprocessor.fit_transform(features)

print("Feature matrix shape:", X.shape)

# Train K-Means clustering model
N_CLUSTERS = 6

print(f"Training K-Means with {N_CLUSTERS} clusters...")
model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

df["cluster"] = model.fit_predict(X)

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

# Save title-level cluster assignments
display_cols = [
    col for col in [
        "show_id", "title", "type", "release_year",
        "rating", "listed_in", "country", "cluster"
    ]
    if col in df.columns
]

df[display_cols].to_csv(
    os.path.join(OUTPUT_DIR, "task4_cluster_assignments.csv"),
    index=False
)

# Summarize each cluster
summary = df.groupby("cluster").agg(
    title_count=("title", "count"),
    average_release_year=("release_year", "mean")
).reset_index()

if "type" in df.columns:
    type_summary = pd.crosstab(
        df["cluster"], df["type"], normalize="index"
    ).reset_index()

    summary = summary.merge(type_summary, on="cluster", how="left")

summary.to_csv(
    os.path.join(OUTPUT_DIR, "task4_cluster_summary.csv"),
    index=False
)

print("\nCluster summary:")
print(summary.to_string(index=False))

# Reduce features to 2 dimensions for visualization
print("\nCreating cluster visualization...")
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=df["cluster"],
    cmap="tab10",
    s=12,
    alpha=0.65
)

plt.title("Netflix Content Segmentation Using K-Means")
plt.xlabel("Metadata Component 1")
plt.ylabel("Metadata Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "task4_clusters.png"),
    dpi=200
)
plt.close()

print("\nTask 4 completed successfully!")
print("Files saved in the outputs folder:")
print("- task4_cluster_assignments.csv")
print("- task4_cluster_summary.csv")
print("- task4_clusters.png")
