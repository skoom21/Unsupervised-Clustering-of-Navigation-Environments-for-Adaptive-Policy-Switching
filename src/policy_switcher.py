from pathlib import Path
from typing import Dict, List

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

import config
from src.data_loader import extract_features, parse_map_file
from src.label_generator import extract_encoded_metadata
from src.rl_agent import GridEnvironment, QAgent, downsample_grid_adaptive, find_canonical_goal

np.random.seed(42)


def _build_feature_row(grid: np.ndarray, map_name: str, feature_columns: List[str]) -> pd.DataFrame:
    features = extract_features(grid, map_name)
    metadata = extract_encoded_metadata(map_name)

    row = dict(features)
    row["corridor_width"] = metadata.get("corridor_width") or 0
    row["num_rooms"] = metadata.get("num_rooms") or 0
    row["known_obstacle_pct"] = metadata.get("known_obstacle_pct")
    row["map_resolution"] = metadata.get("map_resolution") or 512

    if row["known_obstacle_pct"] is None:
        row["known_obstacle_pct"] = 0.0

    data = {col: row.get(col, 0) for col in feature_columns}
    return pd.DataFrame([data])


class AdaptivePolicySwitcher:
    def __init__(
        self,
        kmeans_model,
        scaler,
        pca_reducer,
        cluster_agents: Dict[int, QAgent],
        feature_columns: List[str],
    ) -> None:
        self.kmeans = kmeans_model
        self.scaler = scaler
        self.pca = pca_reducer
        self.agents = cluster_agents
        self.feature_columns = feature_columns

    def identify_cluster(self, grid: np.ndarray, map_name: str = "query_map") -> int:
        features_df = _build_feature_row(grid, map_name, self.feature_columns)
        X_scaled = self.scaler.transform(features_df)
        X_pca = self.pca.transform(X_scaled)
        cluster_id = int(self.kmeans.predict(X_pca)[0])
        logger.info(f"[Switcher] Map assigned to Cluster {cluster_id}")
        return cluster_id

    def get_policy(self, grid: np.ndarray, map_name: str = "query_map") -> QAgent:
        cluster_id = self.identify_cluster(grid, map_name)
        agent = self.agents.get(cluster_id)
        if agent is None:
            raise ValueError(f"No agent available for cluster {cluster_id}")
        return agent

    def run_episode(self, grid: np.ndarray, map_name: str, max_steps: int = 500) -> Dict[str, object]:
        cluster_id = self.identify_cluster(grid, map_name)
        agent = self.agents.get(cluster_id)
        if agent is None:
            raise ValueError(f"No agent available for cluster {cluster_id}")

        metadata = extract_encoded_metadata(map_name)
        if metadata.get("corridor_width") is None:
            metadata["corridor_width"] = 0
        if metadata.get("num_rooms") is None:
            metadata["num_rooms"] = 0

        eval_grid = downsample_grid_adaptive(grid, metadata)
        free_cells = [(r, c) for r in range(eval_grid.shape[0]) 
                            for c in range(eval_grid.shape[1]) if eval_grid[r, c] == 0]
        if not free_cells:
            return {"success": False, "total_reward": -500, "steps": max_steps, "path": [], "cluster_id": cluster_id, "error": "no_free_cells"}
            
        inference_goal = find_canonical_goal(free_cells, eval_grid.shape)
        env = GridEnvironment(eval_grid, max_steps=max_steps, canonical_goal=inference_goal)
        state = env.reset()
        path = [state]
        total_reward = 0.0
        done = False
        while not done:
            action = agent.get_greedy_action(state)
            state, reward, done = env.step(action)
            path.append(state)
            total_reward += reward
        success = (state == env.goal)
        return {
            "success": success,
            "total_reward": total_reward,
            "steps": env.steps,
            "path": path,
            "cluster_id": cluster_id,
        }


def visualize_policy_path(grid: np.ndarray, result_dict: Dict[str, object], map_name: str) -> None:
    path = result_dict.get("path", [])
    if not path:
        logger.warning(f"No path to visualize for {map_name}")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="binary")

    rows = [pos[0] for pos in path]
    cols = [pos[1] for pos in path]
    ax.plot(cols, rows, color="orange", linewidth=2)

    start = path[0]
    goal = path[-1]
    ax.scatter([start[1]], [start[0]], color="green", s=40, label="Start")
    ax.scatter([goal[1]], [goal[0]], color="red", s=60, marker="*", label="Goal")

    ax.set_title(
        f"Cluster {result_dict.get('cluster_id')} | {map_name} | Steps: {result_dict.get('steps')}"
    )
    ax.axis("off")
    ax.legend(loc="upper right")

    safe_name = map_name.replace(".map", "")
    fig_path = config.FIGURES_DIR / f"policy_path_{safe_name}.png"
    try:
        fig.savefig(fig_path, dpi=200)
        logger.info(f"Saved policy path to {fig_path}")
    except OSError as exc:
        logger.warning(f"Failed to save policy path: {exc}")
    finally:
        plt.close(fig)


def _run_agent_episodes(env: GridEnvironment, agent: QAgent, n_episodes: int) -> Dict[str, float]:
    rewards = []
    steps = []
    successes = []
    for _ in range(n_episodes):
        try:
            state = env.reset()
        except ValueError:
            break
        done = False
        total_reward = 0.0
        while not done:
            action = agent.get_greedy_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        steps.append(env.steps)
        successes.append(int(env.agent_pos == env.goal))

    if not rewards:
        return {"success_rate": 0.0, "mean_reward": float("nan"), "mean_steps": float("nan")}

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
    }


def benchmark_adaptive_vs_single(
    test_map_names: List[str],
    df_labeled: pd.DataFrame,
    data_dir: Path,
    switcher: AdaptivePolicySwitcher,
    single_agent: QAgent,
    n_episodes: int = 50,
) -> pd.DataFrame:
    results = []
    data_dir = Path(data_dir)
    map_paths = {path.name: path for path in data_dir.rglob("*.map")}

    for map_name in test_map_names:
        map_row = df_labeled[df_labeled["map_name"] == map_name]
        map_type = str(map_row.iloc[0]["label_map_type"]) if not map_row.empty else "unknown"

        map_path = map_paths.get(map_name)
        if map_path is None:
            logger.warning(f"Map not found on disk: {map_name}")
            continue

        try:
            grid = parse_map_file(map_path)
        except Exception as exc:
            logger.warning(f"Failed to parse {map_name}: {exc}")
            continue

        map_metadata = map_row.iloc[0].to_dict()
        if pd.isna(map_metadata.get("corridor_width")):
            map_metadata["corridor_width"] = 0
        if pd.isna(map_metadata.get("num_rooms")):
            map_metadata["num_rooms"] = 0

        eval_grid = downsample_grid_adaptive(grid, map_metadata)
        free_cells = [(r, c) for r in range(eval_grid.shape[0]) 
                            for c in range(eval_grid.shape[1]) if eval_grid[r, c] == 0]
        if not free_cells:
            logger.warning(f"No free cells in downsampled grid for {map_name}")
            continue
            
        inference_goal = find_canonical_goal(free_cells, eval_grid.shape)
        adaptive_agent = switcher.get_policy(grid, map_name)

        adaptive_env = GridEnvironment(eval_grid, max_steps=config.RL_MAX_STEPS, canonical_goal=inference_goal)
        single_env = GridEnvironment(eval_grid, max_steps=config.RL_MAX_STEPS, canonical_goal=inference_goal)

        adaptive_metrics = _run_agent_episodes(adaptive_env, adaptive_agent, n_episodes)
        single_metrics = _run_agent_episodes(single_env, single_agent, n_episodes)

        results.append(
            {
                "map_name": map_name,
                "map_type": map_type,
                "adaptive_success_rate": adaptive_metrics["success_rate"],
                "adaptive_mean_reward": adaptive_metrics["mean_reward"],
                "adaptive_mean_steps": adaptive_metrics["mean_steps"],
                "single_success_rate": single_metrics["success_rate"],
                "single_mean_reward": single_metrics["mean_reward"],
                "single_mean_steps": single_metrics["mean_steps"],
            }
        )

    df_results = pd.DataFrame(results)
    report_path = config.REPORTS_DIR / "adaptive_vs_single.csv"
    try:
        df_results.to_csv(report_path, index=False)
        logger.info(f"Saved adaptive vs single report to {report_path}")
    except OSError as exc:
        logger.warning(f"Failed to save adaptive vs single report: {exc}")

    return df_results
