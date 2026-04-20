from pathlib import Path
import re
from typing import Dict

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

import config

np.random.seed(42)


def generate_map_type_label(map_name: str) -> str:
    name = map_name.lower()
    if "maze" in name:
        return "maze"
    if "room" in name:
        return "room"
    if "random" in name:
        return "random"
    return "street"


def extract_encoded_metadata(map_name: str) -> Dict[str, object]:
    name = map_name
    lower = map_name.lower()

    corridor_width = None
    num_rooms = None
    known_obstacle_pct = None
    city_name = None
    map_resolution = None

    maze_match = re.match(r"maze\d+-(\d+)-\d+\.map", lower)
    room_match = re.match(r"(\d+)room_\d+\.map", lower)
    random_match = re.match(r"random\d+-(\d+)-\d+\.map", lower)
    street_match = re.match(r"([A-Za-z]+)_[^_]+_(\d+)\.map", name)

    if maze_match:
        corridor_width = int(maze_match.group(1))
    if room_match:
        num_rooms = int(room_match.group(1))
    if random_match:
        known_obstacle_pct = int(random_match.group(1)) / 100.0
    if street_match:
        city_name = street_match.group(1)
        map_resolution = int(street_match.group(2))

    return {
        "corridor_width": corridor_width,
        "num_rooms": num_rooms,
        "known_obstacle_pct": known_obstacle_pct,
        "city_name": city_name,
        "map_resolution": map_resolution,
    }


def compute_difficulty_scores(df: pd.DataFrame) -> pd.Series:
    score = (
        df["obstacle_density"] * 0.40
        + (1.0 / (df["avg_component_size"] + 1)) * 0.30
        + (1.0 / (df["largest_free_component"] + 1)) * 0.30
    )
    return score


def bin_difficulty(scores: pd.Series) -> pd.Series:
    thresholds = scores.quantile([0.33, 0.67])
    p33 = thresholds.iloc[0]
    p67 = thresholds.iloc[1]

    def _bin(score: float) -> str:
        if score <= p33:
            return "easy"
        if score <= p67:
            return "medium"
        return "hard"

    labels = scores.apply(_bin)
    logger.info(f"Difficulty bins: {labels.value_counts().to_dict()}")
    return labels


def generate_density_category(obstacle_density: float) -> str:
    if obstacle_density < config.DENSITY_LOW_THRESHOLD:
        return "low"
    if obstacle_density < config.DENSITY_HIGH_THRESHOLD:
        return "medium"
    return "high"


def generate_all_labels(df_features: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== PHASE 1B: Generating labels ===")
    df = df_features.copy()

    df["label_map_type"] = df["map_name"].apply(generate_map_type_label)
    df["label_difficulty"] = bin_difficulty(compute_difficulty_scores(df))
    df["label_density_category"] = df["obstacle_density"].apply(generate_density_category)

    metadata = df["map_name"].apply(extract_encoded_metadata)
    meta_df = pd.DataFrame(metadata.tolist())

    df["corridor_width"] = meta_df["corridor_width"].fillna(0).astype(int)
    df["num_rooms"] = meta_df["num_rooms"].fillna(0).astype(int)
    df["known_obstacle_pct"] = meta_df["known_obstacle_pct"].astype(float)
    df["city_name"] = meta_df["city_name"].fillna("N/A")
    df["map_resolution"] = meta_df["map_resolution"].fillna(512).astype(int)

    map_type_counts = df["label_map_type"].value_counts().to_dict()
    difficulty_counts = df["label_difficulty"].value_counts().to_dict()
    density_counts = df["label_density_category"].value_counts().to_dict()
    logger.info("=== Label Distribution Summary ===")
    logger.info(f"Map type: {map_type_counts}")
    logger.info(f"Difficulty: {difficulty_counts}")
    logger.info(f"Density category: {density_counts}")

    random_subset = df[df["label_map_type"] == "random"].copy()
    if random_subset["known_obstacle_pct"].notnull().any():
        corr = random_subset[["known_obstacle_pct", "obstacle_density"]].corr().iloc[0, 1]
        logger.info(f"[CHECK] Random map obstacle correlation: r={corr:.3f}")

    output_path = config.PROCESSED_DATA_DIR / "labeled_dataset.csv"
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved labeled dataset to {output_path}")
    except OSError as exc:
        logger.warning(f"Failed to save labeled dataset: {exc}")

    return df


def validate_labels(df_labeled: pd.DataFrame) -> None:
    logger.info("=== PHASE 1B: Validating labels ===")
    lines = []

    def _check(condition: pd.Series, label: str) -> str:
        if condition.empty:
            return f"[WARNING] {label}: no samples"
        pct = 100.0 * condition.mean()
        status = "PASS" if pct >= 70.0 else "WARNING"
        return f"[{status}] {label}: {pct:.2f}%"

    random_low = df_labeled[
        (df_labeled["label_map_type"] == "random")
        & (df_labeled["known_obstacle_pct"] <= 0.15)
    ]
    lines.append(
        _check(
            random_low["label_density_category"] == "low",
            "Random maps (<=0.15) classified low",
        )
    )

    random_high = df_labeled[
        (df_labeled["label_map_type"] == "random")
        & (df_labeled["known_obstacle_pct"] >= 0.35)
    ]
    lines.append(
        _check(
            random_high["label_density_category"] == "high",
            "Random maps (>=0.35) classified high",
        )
    )

    maze_narrow = df_labeled[
        (df_labeled["label_map_type"] == "maze")
        & (df_labeled["corridor_width"] == 1)
    ]
    lines.append(
        _check(
            maze_narrow["label_difficulty"] == "hard",
            "Maze corridor width 1 classified hard",
        )
    )

    maze_wide = df_labeled[
        (df_labeled["label_map_type"] == "maze")
        & (df_labeled["corridor_width"] == 32)
    ]
    lines.append(
        _check(
            maze_wide["label_difficulty"] == "easy",
            "Maze corridor width 32 classified easy",
        )
    )

    random_subset = df_labeled[df_labeled["label_map_type"] == "random"]
    if random_subset["known_obstacle_pct"].notnull().any():
        corr = random_subset[["known_obstacle_pct", "obstacle_density"]].corr().iloc[0, 1]
        status = "PASS" if corr >= 0.90 else "WARNING"
        lines.append(f"[{status}] Random density correlation r={corr:.3f}")
    else:
        lines.append("[WARNING] Random density correlation not computed")

    for line in lines:
        logger.info(line)

    report_path = config.REPORTS_DIR / "label_validation.txt"
    try:
        with report_path.open("w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(lines))
        logger.info(f"Saved validation report to {report_path}")
    except OSError as exc:
        logger.warning(f"Failed to write validation report: {exc}")
