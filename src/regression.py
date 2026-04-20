from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

import config
from src.data_loader import parse_map_file

np.random.seed(42)


def bfs(grid: np.ndarray, start: tuple, goal: tuple, max_nodes: int = 10000) -> int:
    """Returns shortest path length or -1 if unreachable. 4-directional."""
    height, width = grid.shape
    queue = deque([(start, 0)])
    visited = {start}
    visited_count = 0
    while queue:
        if visited_count > max_nodes:
            return -1
            
        (row, col), dist = queue.popleft()
        if (row, col) == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
        visited_count += 1
    return -1


def _index_map_paths(data_dir: Path) -> Dict[str, Path]:
    paths = {}
    for path in data_dir.rglob("*.map"):
        paths[path.name] = path
    return paths


def compute_path_lengths(data_dir: Path, df_features: pd.DataFrame, n_samples: int = 500) -> pd.Series:
    print("=== PHASE 5: Computing BFS path lengths ===")
    map_paths = _index_map_paths(data_dir)
    rng = np.random.default_rng(config.RANDOM_STATE)
    results = []

    for map_name in tqdm(df_features["map_name"].tolist(), desc="BFS", unit="map"):
        map_path = map_paths.get(map_name)
        if map_path is None:
            print(f"[WARNING] Map not found on disk: {map_name}")
            results.append(
                {
                    "map_name": map_name,
                    "avg_path_length": np.nan,
                    "max_path_length": np.nan,
                    "min_path_length": np.nan,
                    "n_reachable_pairs": 0,
                }
            )
            continue

        try:
            grid = parse_map_file(map_path)
        except Exception as exc:
            print(f"[WARNING] Failed to parse {map_name}: {exc}")
            results.append(
                {
                    "map_name": map_name,
                    "avg_path_length": np.nan,
                    "max_path_length": np.nan,
                    "min_path_length": np.nan,
                    "n_reachable_pairs": 0,
                }
            )
            continue

        # Downsample for BFS only to reduce compute while preserving structure.
        grid = grid[::4, ::4]
        free_cells = np.argwhere(grid == 0)
        if free_cells.shape[0] < 10:
            results.append(
                {
                    "map_name": map_name,
                    "avg_path_length": np.nan,
                    "max_path_length": np.nan,
                    "min_path_length": np.nan,
                    "n_reachable_pairs": 0,
                }
            )
            continue

        pair_count = min(n_samples, free_cells.shape[0])
        indices = rng.integers(0, free_cells.shape[0], size=(pair_count, 2))
        same = indices[:, 0] == indices[:, 1]
        while np.any(same):
            indices[same, 1] = rng.integers(0, free_cells.shape[0], size=int(np.sum(same)))
            same = indices[:, 0] == indices[:, 1]

        lengths = []
        for start_idx, goal_idx in indices:
            start = tuple(free_cells[start_idx])
            goal = tuple(free_cells[goal_idx])
            dist = bfs(grid, start, goal)
            if dist >= 0:
                lengths.append(dist)

        if lengths:
            avg_len = float(np.mean(lengths))
            max_len = int(np.max(lengths))
            min_len = int(np.min(lengths))
        else:
            avg_len = np.nan
            max_len = np.nan
            min_len = np.nan

        results.append(
            {
                "map_name": map_name,
                "avg_path_length": avg_len,
                "max_path_length": max_len,
                "min_path_length": min_len,
                "n_reachable_pairs": len(lengths),
            }
        )

    df_results = pd.DataFrame(results)
    output_path = config.PROCESSED_DATA_DIR / "path_lengths.csv"
    try:
        df_results.to_csv(output_path, index=False)
        print(f"[OK] Saved path lengths to {output_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save path lengths: {exc}")

    series = df_results.set_index("map_name")["avg_path_length"]
    series.name = "avg_path_length"
    return series


def train_regression_models(X_train: np.ndarray, y_train: np.ndarray, target_name: str = "target") -> Tuple:
    print(f"=== Training regression models ({target_name}) ===")
    linear = LinearRegression()
    poly = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )

    try:
        linear.fit(X_train, y_train)
        poly.fit(X_train, y_train)
    except Exception as exc:
        print(f"[ERROR] Regression training failed: {exc}")
        raise

    try:
        joblib.dump(linear, config.MODELS_DIR / f"linear_regression_{target_name}.pkl")
        joblib.dump(poly, config.MODELS_DIR / f"poly_regression_{target_name}.pkl")
        print("[OK] Saved regression models")
    except OSError as exc:
        print(f"[WARNING] Failed to save regression models: {exc}")

    return linear, poly


def evaluate_regression(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
    print(f"=== Evaluating regression model: {model_name} ===")
    try:
        y_pred = model.predict(X_test)
    except Exception as exc:
        print(f"[ERROR] Regression prediction failed: {exc}")
        raise

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.7)
    min_val = float(min(np.min(y_test), np.min(y_pred)))
    max_val = float(max(np.max(y_test), np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted: {model_name}")
    fig.tight_layout()
    fig_path = config.FIGURES_DIR / f"regression_actual_vs_pred_{model_name}.png"
    try:
        fig.savefig(fig_path, dpi=200)
        print(f"[OK] Saved actual-vs-pred plot to {fig_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save actual-vs-pred plot: {exc}")
    finally:
        plt.close(fig)

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residuals: {model_name}")
    fig.tight_layout()
    fig_path = config.FIGURES_DIR / f"regression_residuals_{model_name}.png"
    try:
        fig.savefig(fig_path, dpi=200)
        print(f"[OK] Saved residual plot to {fig_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save residual plot: {exc}")
    finally:
        plt.close(fig)

    return {"rmse": rmse, "mae": mae, "r2": r2}
