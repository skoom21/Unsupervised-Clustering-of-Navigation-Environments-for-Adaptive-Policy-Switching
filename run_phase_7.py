from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import config
from src.preprocessing import prepare_numeric_features, handle_missing_values
from src.data_loader import parse_map_file
from src.policy_switcher import AdaptivePolicySwitcher, visualize_policy_path, benchmark_adaptive_vs_single
from src.rl_agent import QAgent, downsample_grid

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
            print(f"[WARNING] Failed to create directory {path}: {exc}")


def _load_labeled_dataset() -> pd.DataFrame:
    labeled_path = config.PROCESSED_DATA_DIR / "labeled_dataset.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {labeled_path}")
    try:
        return pd.read_csv(labeled_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read labeled dataset: {exc}")
        raise


def _load_model(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[ERROR] Failed to load {label}: {exc}")
        raise


def _load_training_maps() -> set[str]:
    report_path = config.REPORTS_DIR / "rl_cluster_metrics.csv"
    if not report_path.exists():
        return set()
    try:
        df = pd.read_csv(report_path)
        return set(df["map_name"].dropna().tolist())
    except Exception as exc:
        print(f"[WARNING] Failed to read RL cluster metrics: {exc}")
        return set()


def _load_cluster_agents(cluster_labels: np.ndarray) -> dict:
    agents = {}
    for cluster_id in sorted(np.unique(cluster_labels)):
        q_path = config.MODELS_DIR / f"qtable_cluster_{cluster_id}.npy"
        if not q_path.exists():
            print(f"[WARNING] Missing Q-table for cluster {cluster_id}: {q_path}")
            continue
        try:
            q_table = np.load(q_path)
            agent = QAgent(q_table.shape[:2])
            agent.Q = q_table
            agent.epsilon = 0.0
            agents[int(cluster_id)] = agent
        except Exception as exc:
            print(f"[WARNING] Failed to load Q-table for cluster {cluster_id}: {exc}")
            continue
    return agents


def _select_single_agent(cluster_labels: np.ndarray, agents: dict) -> QAgent:
    unique, counts = np.unique(cluster_labels, return_counts=True)
    order = [cid for _, cid in sorted(zip(counts, unique), reverse=True)]
    for cluster_id in order:
        agent = agents.get(int(cluster_id))
        if agent is not None:
            print(f"[INFO] Using cluster {cluster_id} as single-agent baseline")
            return agent
    raise ValueError("No available agents for single-agent baseline")


def _select_test_maps(
    df_labeled: pd.DataFrame,
    exclude: set[str],
    n_total: int = 10,
    per_type: int = 2,
) -> list[str]:
    rng = np.random.default_rng(config.RANDOM_STATE)
    selected: list[str] = []

    for map_type in config.MAP_TYPE_CLASSES:
        candidates = df_labeled[
            (df_labeled["label_map_type"] == map_type)
            & (~df_labeled["map_name"].isin(exclude))
        ]["map_name"].tolist()
        if not candidates:
            continue
        count = min(per_type, len(candidates))
        picks = rng.choice(candidates, size=count, replace=False).tolist()
        selected.extend(picks)

    if len(selected) >= n_total:
        return selected[:n_total]

    remaining_pool = df_labeled[
        (~df_labeled["map_name"].isin(exclude))
        & (~df_labeled["map_name"].isin(selected))
    ]["map_name"].tolist()
    if remaining_pool:
        extra_count = min(n_total - len(selected), len(remaining_pool))
        extra = rng.choice(remaining_pool, size=extra_count, replace=False).tolist()
        selected.extend(extra)

    return selected


def run_phase_7() -> None:
    print("\n" + "=" * 60)
    print("PHASE 7: Adaptive Policy Switching")
    print("=" * 60)

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
        print(f"[ERROR] Failed to transform features: {exc}")
        raise

    try:
        cluster_labels = kmeans.predict(X_pca)
    except Exception as exc:
        print(f"[ERROR] Failed to predict clusters: {exc}")
        raise

    cluster_agents = _load_cluster_agents(cluster_labels)
    if not cluster_agents:
        raise ValueError("No cluster agents available; run Phase 6 first")

    feature_columns = X_numeric.columns.tolist()
    switcher = AdaptivePolicySwitcher(kmeans, scaler, pca, cluster_agents, feature_columns)

    single_agent = _select_single_agent(cluster_labels, cluster_agents)
    training_maps = _load_training_maps()
    test_map_names = _select_test_maps(df_labeled, training_maps, n_total=10, per_type=2)
    print(f"[INFO] Phase 7 test maps: {test_map_names}")

    for map_name in test_map_names:
        map_path = config.RAW_DATA_DIR / map_name
        if not map_path.exists():
            map_path = None
            for candidate in config.RAW_DATA_DIR.rglob(map_name):
                map_path = candidate
                break
        if map_path is None:
            print(f"[WARNING] Map not found on disk: {map_name}")
            continue

        try:
            grid = parse_map_file(map_path)
        except Exception as exc:
            print(f"[WARNING] Failed to parse {map_name}: {exc}")
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

    print("[OK] Phase 7 adaptive switching complete")


def main() -> None:
    try:
        run_phase_7()
    except Exception as exc:
        print(f"[ERROR] Phase 7 failed: {exc}")


if __name__ == "__main__":
    main()
