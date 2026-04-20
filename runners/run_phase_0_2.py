import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from pathlib import Path

import logging
import pandas as pd

logger = logging.getLogger(__name__)

import config
from src.setup_utils import extract_all_zips
from src.data_loader import load_all_maps, visualize_sample_maps
from src.label_generator import generate_all_labels, validate_labels
from src.preprocessing import (
    prepare_numeric_features,
    handle_missing_values,
    normalize_features,
    reduce_dimensions,
    create_all_splits,
)


def run_phase_0() -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 0: Setup & Extraction")
    logger.info("=" * 60)
    extract_all_zips(config.DATASET_DIR, config.RAW_DATA_DIR)


def run_phase_1a() -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1A: Feature Extraction")
    logger.info("=" * 60)
    df_features = load_all_maps(config.RAW_DATA_DIR)
    visualize_sample_maps(config.RAW_DATA_DIR)
    return df_features


def run_phase_1b(df_features: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1B: Label Generation")
    logger.info("=" * 60)
    df_labeled = generate_all_labels(df_features)
    validate_labels(df_labeled)
    return df_labeled


def run_phase_2(df_labeled: pd.DataFrame) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Preprocessing")
    logger.info("=" * 60)
    X_numeric, labels_dict = prepare_numeric_features(df_labeled)
    X_clean = handle_missing_values(X_numeric)
    X_scaled, _ = normalize_features(X_clean, fit=True)
    _ = reduce_dimensions(X_scaled, n_components=config.PCA_COMPONENTS)
    _ = create_all_splits(X_scaled, labels_dict)
    logger.info("Phase 2 preprocessing complete")


def main() -> None:
    df_features = None
    df_labeled = None

    try:
        run_phase_0()
    except Exception as exc:
        logger.error(f"Phase 0 failed: {exc}")

    try:
        df_features = run_phase_1a()
    except Exception as exc:
        logger.error(f"Phase 1A failed: {exc}")

    if df_features is not None:
        try:
            df_labeled = run_phase_1b(df_features)
        except Exception as exc:
            logger.error(f"Phase 1B failed: {exc}")

    if df_labeled is not None:
        try:
            run_phase_2(df_labeled)
        except Exception as exc:
            logger.error(f"Phase 2 failed: {exc}")

    logger.info("\n=== PHASE 0-2 RUN COMPLETE ===")


if __name__ == "__main__":
    main()
