import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from pathlib import Path

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import config
from src.preprocessing import prepare_numeric_features, handle_missing_values
from src.regression import compute_path_lengths, train_regression_models, evaluate_regression

np.random.seed(42)


def _ensure_output_dirs() -> None:
    for path in [config.OUTPUTS_DIR, config.FIGURES_DIR, config.MODELS_DIR, config.REPORTS_DIR]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Failed to create directory {path}: {exc}")


def _load_labeled_dataset() -> pd.DataFrame:
    labeled_path = config.PROCESSED_DATA_DIR / "labeled_dataset.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {labeled_path}")
    try:
        return pd.read_csv(labeled_path)
    except Exception as exc:
        logger.error(f"Failed to read labeled dataset: {exc}")
        raise


def _load_or_compute_path_lengths(df_labeled: pd.DataFrame) -> pd.Series:
    path_file = config.PROCESSED_DATA_DIR / "path_lengths.csv"
    if path_file.exists():
        try:
            df_paths = pd.read_csv(path_file)
            if "map_name" in df_paths.columns and "avg_path_length" in df_paths.columns:
                series = df_paths.set_index("map_name")["avg_path_length"]
                series.name = "avg_path_length"
                logger.info(f"Loaded path lengths from {path_file}")
                return series
        except Exception as exc:
            logger.warning(f"Failed to read path lengths: {exc}")

    return compute_path_lengths(config.RAW_DATA_DIR, df_labeled, n_samples=config.BFS_SAMPLES)


def _save_phase5_scaler(scaler: StandardScaler) -> None:
    scaler_path = config.MODELS_DIR / "scaler_phase5.pkl"
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved Phase 5 scaler to {scaler_path}")
    except OSError as exc:
        logger.warning(f"Failed to save Phase 5 scaler: {exc}")


def run_phase_5() -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5: Regression (BFS)")
    logger.info("=" * 60)

    _ensure_output_dirs()
    df_labeled = _load_labeled_dataset()

    path_series = _load_or_compute_path_lengths(df_labeled)
    df_labeled = df_labeled.merge(path_series, left_on="map_name", right_index=True, how="left")

    X_numeric, labels_dict = prepare_numeric_features(df_labeled)
    # Avoid leakage by removing BFS-derived targets from features.
    leakage_cols = ["avg_path_length", "max_path_length", "min_path_length", "n_reachable_pairs"]
    X_numeric = X_numeric.drop(columns=[col for col in leakage_cols if col in X_numeric.columns])
    X_clean = handle_missing_values(X_numeric)

    if "avg_path_length" not in df_labeled.columns:
        raise ValueError("avg_path_length target not found; BFS computation failed")

    target = df_labeled["avg_path_length"]
    mask = target.notna()
    X_filtered = X_clean.loc[mask]
    y_filtered = target.loc[mask].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered,
        y_filtered,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    _save_phase5_scaler(scaler)

    linear_model, poly_model = train_regression_models(
        X_train_scaled, y_train, target_name="avg_path_length"
    )

    metrics = {}
    metrics["linear"] = evaluate_regression(
        linear_model, X_test_scaled, y_test, "linear_avg_path_length"
    )
    metrics["polynomial"] = evaluate_regression(
        poly_model, X_test_scaled, y_test, "poly_avg_path_length"
    )

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.index.name = "model"
    report_path = config.REPORTS_DIR / "regression_avg_path_length_metrics.csv"
    try:
        metrics_df.to_csv(report_path)
        logger.info(f"Saved regression metrics to {report_path}")
    except OSError as exc:
        logger.warning(f"Failed to save regression metrics: {exc}")

    logger.info("Phase 5 regression complete")


def main() -> None:
    try:
        run_phase_5()
    except Exception as exc:
        logger.error(f"Phase 5 failed: {exc}")


if __name__ == "__main__":
    main()
