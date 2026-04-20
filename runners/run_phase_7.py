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
import joblib

import config
from src.preprocessing import prepare_numeric_features, handle_missing_values
from src.data_loader import parse_map_file
from src.policy_switcher import AdaptivePolicySwitcher, visualize_policy_path, benchmark_adaptive_vs_single
from src.rl_agent import QAgent, downsample_grid, _train_cluster_worker

np.random.seed(42)


def _ensure_output_dirs() -> None:
    for path in [
        config.OUTPUTS_DIR,
        config.FIGURES_DIR,
        config.MODELS_DIR,
        config.REPORTS_DIR,
        config.PROCESSED_DATA_DIR,
    ]:
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


def _load_model(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.error(f"Failed to load {label}: {exc}")
        raise


def _load_training_maps() -> set[str]:
    report_path = config.REPORTS_DIR / "rl_cluster_metrics.csv"
    if not report_path.exists():
        return set()
    try:
        df = pd.read_csv(report_path)
        return set(df["map_name"].dropna().tolist())
    except Exception as exc:
        logger.warning(f"Failed to read RL cluster metrics: {exc}")
        return set()


def _load_cluster_agents(cluster_labels: np.ndarray) -> dict:
    agents = {}
    for cluster_id in sorted(np.unique(cluster_labels)):
        q_path = config.MODELS_DIR / f"qtable_cluster_{cluster_id}.npy"
        goal_path = config.MODELS_DIR / f"canonical_goal_cluster_{cluster_id}.npy"
        if not q_path.exists() or not goal_path.exists():
            logger.warning(f"Missing Q-table or goal for cluster {cluster_id}")
            continue
        try:
            q_table = np.load(q_path)
            agent = QAgent(q_table.shape[:2])
            agent.Q = q_table
            agent.epsilon = 0.0
            agent.canonical_goal = tuple(np.load(goal_path).tolist())
            agents[int(cluster_id)] = agent
        except Exception as exc:
            logger.warning(f"Failed to load Q-table for cluster {cluster_id}: {exc}")
            continue
    return agents


def _train_single_baseline_agent(df_labeled, X_pca, data_dir, config) -> QAgent:
    """Train one agent on the map closest to the global centroid."""
    global_centroid = X_pca.mean(axis=0)
    distances = np.linalg.norm(X_pca - global_centroid, axis=1)
    closest_idx = np.argmin(distances)
    baseline_map = df_labeled.iloc[closest_idx]["map_name"]
    logger.info(f"Baseline agent training on: {baseline_map} (closest to global centroid)")
    
    # Train using same procedure as cluster agents
    map_features = df_labeled.iloc[closest_idx].to_dict()
    map_path = list(Path(data_dir).rglob(baseline_map))
    if not map_path:
        raise FileNotFoundError(f"Could not find baseline map: {baseline_map}")
        
    task = {
        "cluster_id": -1,
        "map_name": baseline_map,
        "map_path": str(map_path[0]),
        "map_type": str(map_features.get("label_map_type", "unknown")),
        "map_features": map_features
    }
    
    result = _train_cluster_worker(task)
    if result.get("status") != "ok":
        raise RuntimeError("Failed to train baseline agent")
        
    q_path = Path(result["qtable_path"])
    q_table = np.load(q_path)
    agent = QAgent(q_table.shape[:2])
    agent.Q = q_table
    agent.epsilon = 0.0
    agent.canonical_goal = tuple(np.load(result["goal_path"]).tolist())
    return agent


TRAINING_MAPS = {
    "Shanghai_2_512.map",
    "Moscow_1_1024.map", 
    "maze512-8-9.map",
    "random512-35-6.map"
}

def _select_test_maps(df_labeled, n_per_type=2):
    """Select 2 unseen maps per type for benchmarking."""
    test_maps = []
    for map_type in ["maze", "room", "random", "street"]:
        candidates = df_labeled[
            (df_labeled["label_map_type"] == map_type) &
            (~df_labeled["map_name"].isin(TRAINING_MAPS)) &
            (df_labeled["corridor_width"] != 1) &
            (df_labeled["free_ratio"] > 0.3)
        ]
        selected = candidates.sample(n=min(n_per_type, len(candidates)), 
                                     random_state=42)
        test_maps.extend(selected["map_name"].tolist())
    return test_maps


def run_phase_7() -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 7: Adaptive Policy Switching")
    logger.info("=" * 60)

    _ensure_output_dirs()
    df_labeled = _load_labeled_dataset()

    scaler = _load_model(config.MODELS_DIR / "scaler.pkl", "Scaler")
    pca = _load_model(config.MODELS_DIR / "pca_reducer.pkl", "PCA model")
    kmeans = _load_model(config.MODELS_DIR / "kmeans_model.pkl", "KMeans model")

    X_numeric, _ = prepare_numeric_features(df_labeled)
    X_clean = handle_missing_values(X_numeric)
    try:
        X_scaled = scaler.transform(X_clean)
        X_pca = pca.transform(X_scaled)
    except Exception as exc:
        logger.error(f"Failed to transform features: {exc}")
        raise

    try:
        cluster_labels = kmeans.predict(X_pca)
    except Exception as exc:
        logger.error(f"Failed to predict clusters: {exc}")
        raise

    cluster_agents = _load_cluster_agents(cluster_labels)
    if not cluster_agents:
        raise ValueError("No cluster agents available; run Phase 6 first")

    feature_columns = X_numeric.columns.tolist()
    switcher = AdaptivePolicySwitcher(kmeans, scaler, pca, cluster_agents, feature_columns)

    single_agent = _train_single_baseline_agent(df_labeled, X_pca, config.RAW_DATA_DIR, config)
    test_map_names = _select_test_maps(df_labeled, n_per_type=2)
    logger.info(f"Phase 7 test maps: {test_map_names}")

    for map_name in test_map_names:
        map_path = config.RAW_DATA_DIR / map_name
        if not map_path.exists():
            map_path = None
            for candidate in config.RAW_DATA_DIR.rglob(map_name):
                map_path = candidate
                break
        if map_path is None:
            logger.warning(f"Map not found on disk: {map_name}")
            continue

        try:
            grid = parse_map_file(map_path)
        except Exception as exc:
            logger.warning(f"Failed to parse {map_name}: {exc}")
            continue

        result = switcher.run_episode(grid, map_name, max_steps=config.RL_MAX_STEPS)
        eval_grid = downsample_grid(grid, getattr(config, "RL_MAX_GRID_DIM", None))
        visualize_policy_path(eval_grid, result, map_name)

    benchmark_adaptive_vs_single(
        test_map_names,
        df_labeled,
        config.RAW_DATA_DIR,
        switcher,
        single_agent,
        n_episodes=50,
    )

    logger.info("Phase 7 adaptive switching complete")


def main() -> None:
    try:
        run_phase_7()
    except Exception as exc:
        logger.error(f"Phase 7 failed: {exc}")


if __name__ == "__main__":
    main()
