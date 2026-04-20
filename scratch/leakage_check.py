import pandas as pd
import numpy as np
from pathlib import Path

# Load data
df = pd.read_csv('data/processed/labeled_dataset.csv')
paths = pd.read_csv('data/processed/path_lengths.csv')

# Merge
merged = df.merge(paths[['map_name','avg_path_length']], on='map_name', how='left')
merged = merged.dropna(subset=['avg_path_length'])

# Numeric correlations
numeric_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in ['avg_path_length']]
correlations = merged[numeric_cols].corrwith(merged['avg_path_length']).abs().sort_values(ascending=False)

print('Top 10 correlations:')
print(correlations.head(10))

print('\nMetadata feature correlations:')
target_metadata = ['corridor_width', 'num_rooms', 'known_obstacle_pct', 'map_resolution']
for col in target_metadata:
    if col in merged.columns:
        r = merged[col].corr(merged['avg_path_length'])
        print(f"  {col}: r={r:.3f}")
    else:
        print(f"  {col}: Not found in columns")
