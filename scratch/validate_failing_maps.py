import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from src.data_loader import parse_map_file
import config

def bfs(grid: np.ndarray, start: tuple, goal: tuple) -> int:
    height, width = grid.shape
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, col), dist = queue.popleft()
        if (row, col) == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return -1

def _index_map_paths(data_dir: Path):
    paths = {}
    for path in data_dir.rglob("*.map"):
        paths[path.name] = path
    return paths

# Load maps and path lengths
df = pd.read_csv('data/processed/labeled_dataset.csv')
paths = pd.read_csv('data/processed/path_lengths.csv')
merged = df.merge(paths[['map_name','avg_path_length']], on='map_name', how='left')

# Identify failing maps
failing = merged[merged['avg_path_length'].isna()]
print(f"Checking {len(failing)} failing maps at full resolution...")

map_paths = _index_map_paths(config.RAW_DATA_DIR)
rng = np.random.default_rng(42)

results = []
for map_name in failing['map_name']:
    map_path = map_paths.get(map_name)
    if not map_path:
        continue
    
    grid = parse_map_file(map_path)
    free_cells = np.argwhere(grid == 0)
    
    if free_cells.shape[0] < 2:
        results.append((map_name, 0, "No free space"))
        continue
    
    # Try 100 random pairs at FULL resolution
    reachable_count = 0
    pair_count = min(100, free_cells.shape[0])
    indices = rng.integers(0, free_cells.shape[0], size=(pair_count, 2))
    
    for start_idx, goal_idx in indices:
        start = tuple(free_cells[start_idx])
        goal = tuple(free_cells[goal_idx])
        if start == goal: continue
        if bfs(grid, start, goal) >= 0:
            reachable_count += 1
            
    results.append((map_name, reachable_count, pair_count))

res_df = pd.DataFrame(results, columns=['map_name', 'reachable', 'total'])
res_df['percent'] = (res_df['reachable'] / res_df['total'] * 100).fillna(0)

print("\nFull Resolution Connectivity Results for failing maps:")
print(res_df.sort_values('percent', ascending=False).to_string())

# Summarize by type
failing_with_res = failing.merge(res_df, on='map_name')
print("\nSuccess by map type at full resolution:")
print(failing_with_res[failing_with_res['reachable'] > 0]['label_map_type'].value_counts())
