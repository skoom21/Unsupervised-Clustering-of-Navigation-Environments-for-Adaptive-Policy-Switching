import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/processed/labeled_dataset.csv")
paths = pd.read_csv("data/processed/path_lengths.csv")

# Merge
merged = df.merge(paths[["map_name","avg_path_length"]], on="map_name", how="left")
merged = merged.dropna(subset=["avg_path_length"])

# Define numeric columns for correlation check
numeric_cols = [c for c in merged.select_dtypes(include=[np.number]).columns
                if c != "avg_path_length"]

# Calculate correlations
correlations = merged[numeric_cols].corrwith(merged["avg_path_length"]).abs().sort_values(ascending=False)

print("=== All feature correlations with avg_path_length ===")
print(correlations.to_string())

print("\n=== Metadata feature correlations (un-abs'd) ===")
for col in ["corridor_width", "num_rooms", "known_obstacle_pct", "map_resolution"]:
    if col in merged.columns:
        r = merged[col].corr(merged["avg_path_length"])
        print(f"  {col}: r={r:.3f}")

print("\n=== Train/test type distribution check ===")
X = merged[numeric_cols]
y = merged["avg_path_length"]
# Note: Using random_state=42 as requested by user
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Map indices back to original merged dataframe to get labels
test_labels = merged.iloc[X_test.index]["label_map_type"].value_counts()
print(f"Test set size: {len(X_test)}")
print(f"Test set composition: {test_labels.to_dict()}")
