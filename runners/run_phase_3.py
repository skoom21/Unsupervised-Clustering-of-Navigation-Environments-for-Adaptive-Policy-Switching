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
import config
from src.preprocessing import prepare_numeric_features, handle_missing_values, normalize_features, reduce_dimensions
from src.clustering import (
    add_suffix,
    find_optimal_k,
    run_kmeans,
    evaluate_clustering,
    visualize_clusters_2d,
    cross_tabulate_clusters_vs_labels,
    analyze_cluster_profiles,
    visualize_cluster_samples,
)

np.random.seed(42)


def run_phase_3() -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Clustering")
    logger.info("=" * 60)

    labeled_path = config.PROCESSED_DATA_DIR / "labeled_dataset.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {labeled_path}")

    df_labeled = pd.read_csv(labeled_path)
    X_numeric, _ = prepare_numeric_features(df_labeled)
    X_clean = handle_missing_values(X_numeric)
    X_scaled, _ = normalize_features(X_clean, fit=True)
    X_pca, _ = reduce_dimensions(X_scaled, n_components=config.PCA_COMPONENTS)

    _, recommended_k = find_optimal_k(X_pca)
    best_k = config.BEST_K
    downstream_k = config.DOWNSTREAM_CLUSTER_K

    logger.info(f"Silhouette-optimal k = {recommended_k}")
    logger.info(f"Using k = {downstream_k} for downstream tasks (better semantic separation)")

    X_2d = X_pca[:, :2]

    if best_k != downstream_k:
        suffix = f"_k{best_k}"
        logger.info(f"Saving metric-optimal k={best_k} outputs with suffix '{suffix}'")
        model_opt, labels_opt = run_kmeans(X_pca, n_clusters=best_k, suffix=suffix)
        _ = evaluate_clustering(X_pca, labels_opt, suffix=suffix)
        visualize_clusters_2d(X_2d, labels_opt, suffix=suffix)
        cross_tabulate_clusters_vs_labels(df_labeled, labels_opt, suffix=suffix)
        analyze_cluster_profiles(df_labeled, labels_opt, suffix=suffix)
        visualize_cluster_samples(config.RAW_DATA_DIR, df_labeled, labels_opt, suffix=suffix)
        labels_path = config.MODELS_DIR / f"cluster_labels_k{best_k}.npy"
        try:
            np.save(labels_path, labels_opt)
            logger.info(f"Saved cluster labels to {labels_path}")
        except OSError as exc:
            logger.warning(f"Failed to save cluster labels: {exc}")

    logger.info(f"Using downstream k={downstream_k} for default artifacts")
    model, labels = run_kmeans(X_pca, n_clusters=downstream_k)
    _ = evaluate_clustering(X_pca, labels)
    visualize_clusters_2d(X_2d, labels)
    cross_tabulate_clusters_vs_labels(df_labeled, labels)
    analyze_cluster_profiles(df_labeled, labels)
    visualize_cluster_samples(config.RAW_DATA_DIR, df_labeled, labels)
    labels_path = config.MODELS_DIR / f"cluster_labels_k{downstream_k}.npy"
    try:
        np.save(labels_path, labels)
        logger.info(f"Saved cluster labels to {labels_path}")
    except OSError as exc:
        logger.warning(f"Failed to save cluster labels: {exc}")

    logger.info("Phase 3 clustering complete")


def main() -> None:
    try:
        run_phase_3()
    except Exception as exc:
        logger.error(f"Phase 3 failed: {exc}")


if __name__ == "__main__":
    main()
