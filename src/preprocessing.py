from typing import Dict, Tuple

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

import config

np.random.seed(42)


def prepare_numeric_features(df_labeled: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    logger.info("=== PHASE 2: Preparing numeric features ===")
    df = df_labeled.copy()
    label_cols = ["label_map_type", "label_difficulty", "label_density_category"]

    labels_dict = {col: df[col] for col in label_cols if col in df.columns}
    for target in ["avg_path_length", "avg_steps_to_goal"]:
        if target in df.columns:
            labels_dict[target] = df[target]

    drop_cols = ["map_name", "city_name"] + label_cols
    drop_cols = [col for col in drop_cols if col in df.columns]
    X_numeric = df.drop(columns=drop_cols)

    for col in ["corridor_width", "num_rooms", "known_obstacle_pct", "map_resolution"]:
        if col in X_numeric.columns:
            X_numeric[col] = X_numeric[col].fillna(0)

    logger.info(f"Numeric feature shape: {X_numeric.shape}")
    logger.info(f"Numeric dtypes:\n{X_numeric.dtypes}")
    logger.info(f"Null counts:\n{X_numeric.isnull().sum()}")
    return X_numeric, labels_dict


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== PHASE 2: Handling missing values ===")
    logger.info(f"Null counts before:\n{df.isnull().sum()}")
    logger.info(f"Any inf values: {np.isinf(df.values).any()}")

    df_clean = df.replace([np.inf, -np.inf], np.nan)
    null_counts = df_clean.isnull().sum()
    imputed_cols = null_counts[null_counts > 0].index.tolist()
    for col in imputed_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

    if imputed_cols:
        logger.info(f"Imputed columns: {imputed_cols}")
        logger.info(f"Imputed counts:\n{null_counts[imputed_cols]}")
    else:
        logger.info("No missing values found.")

    return df_clean


def normalize_features(X: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None):
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        try:
            joblib.dump(scaler, config.MODELS_DIR / "scaler.pkl")
            logger.info("Saved scaler to outputs/models/scaler.pkl")
        except OSError as exc:
            logger.warning(f"Failed to save scaler: {exc}")
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def reduce_dimensions(X_scaled: np.ndarray, n_components: int = 10):
    logger.info("=== PHASE 2: PCA reduction ===")
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_reduced = pca.fit_transform(X_scaled)

    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    for idx, (r, c) in enumerate(zip(ratios, cum), start=1):
        logger.info(f"PC{idx}: {r:.4f} (cumulative {c:.4f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(ratios) + 1), ratios, alpha=0.7, label="Variance")
    ax.plot(range(1, len(cum) + 1), cum, marker="o", label="Cumulative")
    ax.axhline(0.9, color="red", linestyle="--", label="90%")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.legend()
    plt.tight_layout()

    fig_path = config.FIGURES_DIR / "pca_variance.png"
    try:
        fig.savefig(fig_path, dpi=200)
        logger.info(f"Saved PCA variance plot to {fig_path}")
    except OSError as exc:
        logger.warning(f"Failed to save PCA plot: {exc}")
    finally:
        plt.close(fig)

    try:
        joblib.dump(pca, config.MODELS_DIR / "pca_reducer.pkl")
        logger.info("Saved PCA model to outputs/models/pca_reducer.pkl")
    except OSError as exc:
        logger.warning(f"Failed to save PCA model: {exc}")

    return X_reduced, pca


def create_all_splits(X_scaled: np.ndarray, labels_dict: Dict[str, pd.Series]):
    logger.info("=== PHASE 2: Creating train/test splits ===")
    splits = {}
    for label_name, y in labels_dict.items():
        stratify = y if label_name.startswith("label_") else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=stratify,
        )
        if stratify is not None:
            logger.info(f"Split {label_name}:")
            logger.info(f"Train distribution: {y_train.value_counts().to_dict()}")
            logger.info(f"Test distribution: {y_test.value_counts().to_dict()}")
        splits[label_name] = (X_train, X_test, y_train, y_test)

    return splits
