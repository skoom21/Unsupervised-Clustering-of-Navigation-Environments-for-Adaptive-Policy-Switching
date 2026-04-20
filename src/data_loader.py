from pathlib import Path
import gc
from typing import Dict, List

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)
from scipy import stats, ndimage
import matplotlib.pyplot as plt

import config

np.random.seed(42)

FREE_CHARS = {".", "G", "S"}
OBSTACLE_CHARS = {"@", "O", "T", "W"}


def parse_map_file(filepath: Path) -> np.ndarray:
    if not filepath.exists():
        raise FileNotFoundError(f"Map file not found: {filepath}")

    try:
        with filepath.open("r", encoding="utf-8", errors="ignore") as file_obj:
            lines = [line.strip() for line in file_obj]
        if len(lines) < 4:
            raise ValueError("Header missing")

        height_parts = lines[1].split()
        width_parts = lines[2].split()
        height = int(height_parts[1])
        width = int(width_parts[1])

        grid_lines = lines[4:]
        if len(grid_lines) != height:
            raise ValueError("Height mismatch")

        grid = np.zeros((height, width), dtype=np.int8)
        for r, line in enumerate(grid_lines):
            if len(line) != width:
                raise ValueError("Width mismatch")
            for c, ch in enumerate(line):
                if ch in FREE_CHARS:
                    grid[r, c] = 0
                elif ch in OBSTACLE_CHARS:
                    grid[r, c] = 1
                else:
                    raise ValueError(f"Unknown cell character: {ch}")

        if grid.shape != (height, width):
            raise ValueError("Declared shape mismatch")

        return grid
    except Exception as exc:
        raise ValueError(f"Malformed map: {filepath.name}") from exc


def _component_sizes(free_mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(free_mask)
    if num == 0:
        return np.array([], dtype=np.int64)
    counts = np.bincount(labeled.ravel())
    if counts.size > 0:
        counts[0] = 0
    return counts[counts > 0]


def _symmetry_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    return float(np.nan_to_num(corr, nan=0.0))


def extract_features(grid: np.ndarray, map_name: str) -> Dict[str, float]:
    obstacle_density = float(grid.mean())
    free_ratio = 1.0 - obstacle_density
    total_cells = int(grid.size)
    free_cells = int((grid == 0).sum())
    obstacle_cells = int((grid == 1).sum())
    height, width = grid.shape
    aspect_ratio = float(width / height)

    values, counts = np.unique(grid, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(stats.entropy(probs))

    free_mask = (1 - grid).astype(np.int8)
    sizes = _component_sizes(free_mask)
    largest_free_component = int(sizes.max()) if sizes.size else 0
    num_free_components = int(sizes.size)
    avg_component_size = float(free_cells / max(num_free_components, 1))
    component_size_variance = float(np.var(sizes)) if sizes.size else 0.0

    top = grid[0, :]
    bottom = grid[-1, :]
    left = grid[1:-1, 0] if height > 2 else grid[:, 0]
    right = grid[1:-1, -1] if width > 2 else grid[:, -1]
    border = np.concatenate([top, bottom, left, right])
    border_obstacle_ratio = float(border.mean()) if border.size else 0.0

    if height % 2 == 0:
        top_half = grid[: height // 2, :]
        bottom_half = grid[height // 2 :, :]
    else:
        top_half = grid[: height // 2, :]
        bottom_half = grid[height // 2 + 1 :, :]
    horizontal_symmetry = _symmetry_score(top_half, np.flipud(bottom_half))

    if width % 2 == 0:
        left_half = grid[:, : width // 2]
        right_half = grid[:, width // 2 :]
    else:
        left_half = grid[:, : width // 2]
        right_half = grid[:, width // 2 + 1 :]
    vertical_symmetry = _symmetry_score(left_half, np.fliplr(right_half))

    return {
        "obstacle_density": obstacle_density,
        "free_ratio": free_ratio,
        "total_cells": total_cells,
        "free_cells": free_cells,
        "obstacle_cells": obstacle_cells,
        "height": height,
        "width": width,
        "aspect_ratio": aspect_ratio,
        "entropy": entropy,
        "largest_free_component": largest_free_component,
        "num_free_components": num_free_components,
        "avg_component_size": avg_component_size,
        "component_size_variance": component_size_variance,
        "border_obstacle_ratio": border_obstacle_ratio,
        "horizontal_symmetry": horizontal_symmetry,
        "vertical_symmetry": vertical_symmetry,
        "map_name": map_name,
    }


def _infer_map_type(map_name: str) -> str:
    name = map_name.lower()
    if "maze" in name:
        return "maze"
    if "room" in name:
        return "room"
    if "random" in name:
        return "random"
    return "street"


def load_all_maps(data_dir: Path) -> pd.DataFrame:
    logger.info("=== PHASE 1A: Loading maps and extracting features ===")
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    map_paths = sorted(data_dir.rglob("*.map"))
    skipped: List[str] = []
    features: List[Dict[str, float]] = []
    counts = {"maze": 0, "room": 0, "random": 0, "street": 0}

    for map_path in map_paths:
        try:
            grid = parse_map_file(map_path)
            height, width = grid.shape
            if height < 10 or width < 10:
                skipped.append(map_path.name)
                continue
            feats = extract_features(grid, map_path.name)
            features.append(feats)
            counts[_infer_map_type(map_path.name)] += 1
            if max(height, width) > 512:
                del grid
                gc.collect()
        except (OSError, ValueError) as exc:
            logger.warning(f"Skipping {map_path.name}: {exc}")
            skipped.append(map_path.name)

    df = pd.DataFrame(features)
    logger.info(f"Loaded maps: {len(df)}")
    logger.info(f"Counts by type: {counts}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Null counts:\n{df.isnull().sum()}")

    if skipped:
        report_path = config.REPORTS_DIR / "skipped_maps.txt"
        try:
            with report_path.open("w", encoding="utf-8") as file_obj:
                file_obj.write("\n".join(skipped))
            logger.info(f"Skipped maps logged to {report_path}")
        except OSError as exc:
            logger.warning(f"Failed to write skipped map report: {exc}")

    output_path = config.PROCESSED_DATA_DIR / "raw_features.csv"
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
    except OSError as exc:
        logger.warning(f"Failed to save features: {exc}")

    return df


def _resolve_type_dirs(data_dir: Path) -> Dict[str, Path]:
    candidates = {
        "maze": ["maze", "maze-map"],
        "room": ["room", "room-map"],
        "random": ["random", "random-map"],
        "street": ["street", "street-map"],
    }
    resolved: Dict[str, Path] = {}
    for key, names in candidates.items():
        for name in names:
            path = data_dir / name
            if path.exists():
                resolved[key] = path
                break
    return resolved


def visualize_sample_maps(data_dir: Path, n_per_type: int = 3) -> None:
    logger.info("=== PHASE 1A: Visualizing sample maps ===")
    data_dir = Path(data_dir)
    type_dirs = _resolve_type_dirs(data_dir)

    if len(type_dirs) < 4:
        logger.warning(f"Missing some map type folders in {data_dir}")

    fig, axes = plt.subplots(4, n_per_type, figsize=(n_per_type * 3, 12))
    for row, map_type in enumerate(["maze", "room", "random", "street"]):
        map_dir = type_dirs.get(map_type)
        if not map_dir:
            continue
        map_files = list(map_dir.glob("*.map"))
        if not map_files:
            continue
        samples = np.random.choice(map_files, size=min(n_per_type, len(map_files)), replace=False)
        for col, map_path in enumerate(samples):
            grid = parse_map_file(map_path)
            feats = extract_features(grid, map_path.name)
            ax = axes[row, col]
            ax.imshow(grid, cmap="binary")
            ax.set_title(f"{map_path.name}\ndensity={feats['obstacle_density']:.2f}")
            ax.axis("off")
        axes[row, 0].set_ylabel(map_type.capitalize())

    plt.tight_layout()
    output_path = config.FIGURES_DIR / "sample_maps_by_type.png"
    try:
        fig.savefig(output_path, dpi=200)
        logger.info(f"Saved sample map figure to {output_path}")
    except OSError as exc:
        logger.warning(f"Failed to save sample map figure: {exc}")
    finally:
        plt.close(fig)
