from pathlib import Path
from typing import Dict, List, Tuple
import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.data_loader import parse_map_file

np.random.seed(42)


class GridEnvironment:
    def __init__(self, grid: np.ndarray, max_steps: int | None = None):
        self.grid = grid
        self.H, self.W = grid.shape
        self.max_steps = max_steps or (self.H * self.W * 2)
        self.free_cells = [(r, c) for r in range(self.H) for c in range(self.W) if grid[r, c] == 0]
        self.agent_pos = None
        self.goal = None
        self.steps = 0

    def reset(self) -> tuple:
        if len(self.free_cells) < 2:
            raise ValueError("Not enough free cells to sample start/goal")
        positions = np.random.choice(len(self.free_cells), size=2, replace=False)
        self.agent_pos = self.free_cells[positions[0]]
        self.goal = self.free_cells[positions[1]]
        self.steps = 0
        return self.agent_pos

    def step(self, action: int) -> tuple:
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        self.steps += 1
        if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] == 0:
            self.agent_pos = (nr, nc)
            reward = -1
        else:
            reward = -5
        done = (self.agent_pos == self.goal) or (self.steps >= self.max_steps)
        if self.agent_pos == self.goal:
            reward = 10
        return self.agent_pos, reward, done


class QAgent:
    def __init__(
        self,
        grid_shape: tuple,
        n_actions: int = 4,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.Q = np.zeros((*grid_shape, n_actions), dtype=np.float32)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state: tuple) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(4))
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.lr * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_greedy_action(self, state: tuple) -> int:
        return int(np.argmax(self.Q[state]))


def train_agent(
    env: GridEnvironment,
    agent: QAgent,
    n_episodes: int = 2000,
    log_every: int = 50,
    step_log_interval: int = 50000,
    early_stop_window: int = 50,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1.0,
) -> Tuple[List[float], List[int], List[int]]:
    rewards: List[float] = []
    steps: List[int] = []
    success: List[int] = []
    best_avg = float("-inf")
    no_improve = 0

    for episode in range(n_episodes):
        try:
            state = env.reset()
        except ValueError as exc:
            print(f"[WARNING] Episode skipped: {exc}")
            break

        total_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if env.steps % step_log_interval == 0:
                print(
                    f"[TRAIN] Episode {episode + 1}/{n_episodes} steps={env.steps} "
                    f"reward={total_reward:.1f} epsilon={agent.epsilon:.4f}"
                )

        rewards.append(total_reward)
        steps.append(env.steps)
        success.append(int(env.agent_pos == env.goal))

        if (episode + 1) % log_every == 0:
            window = min(log_every, len(rewards))
            avg_reward = float(np.mean(rewards[-window:]))
            avg_steps = float(np.mean(steps[-window:]))
            success_rate = float(np.mean(success[-window:]))
            print(
                f"[TRAIN] Episode {episode + 1}/{n_episodes} avg_reward={avg_reward:.2f} "
                f"avg_steps={avg_steps:.1f} success_rate={success_rate:.2f} epsilon={agent.epsilon:.4f}"
            )

            if len(rewards) >= early_stop_window:
                recent_avg = float(np.mean(rewards[-early_stop_window:]))
                if recent_avg > best_avg + early_stop_min_delta:
                    best_avg = recent_avg
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= early_stop_patience:
                        print(
                            f"[TRAIN] Early stopping at episode {episode + 1}: "
                            f"recent_avg={recent_avg:.2f} best_avg={best_avg:.2f}"
                        )
                        break

        agent.decay_epsilon()

    return rewards, steps, success


def evaluate_agent(env: GridEnvironment, agent: QAgent, n_eval: int = 100) -> Dict[str, float]:
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    steps = []
    successes = []
    try:
        for _ in range(n_eval):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = agent.choose_action(state)
                state, reward, done = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
            steps.append(env.steps)
            successes.append(int(env.agent_pos == env.goal))
    except ValueError as exc:
        print(f"[WARNING] Evaluation skipped: {exc}")
    finally:
        agent.epsilon = original_epsilon

    if not rewards:
        return {
            "success_rate": 0.0,
            "mean_cumulative_reward": float("nan"),
            "mean_steps_to_goal": float("nan"),
        }

    return {
        "success_rate": float(np.mean(successes)),
        "mean_cumulative_reward": float(np.mean(rewards)),
        "mean_steps_to_goal": float(np.mean(steps)),
    }


def plot_training_curves(rewards: List[float], steps: List[int], cluster_id: int) -> None:
    if not rewards:
        print(f"[WARNING] No rewards to plot for cluster {cluster_id}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rewards_series = pd.Series(rewards)
    rolling = rewards_series.rolling(50).mean()

    axes[0].plot(rewards, alpha=0.5, label="Reward")
    axes[0].plot(rolling, color="red", label="Rolling mean (50)")
    axes[0].set_title(f"Cluster {cluster_id} Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(steps, alpha=0.7)
    axes[1].set_title(f"Cluster {cluster_id} Steps")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")

    plt.tight_layout()
    fig_path = config.FIGURES_DIR / f"rl_training_cluster_{cluster_id}.png"
    try:
        fig.savefig(fig_path, dpi=200)
        print(f"[OK] Saved training curve to {fig_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save training curve: {exc}")
    finally:
        plt.close(fig)


def select_representative_map(
    df_labeled: pd.DataFrame,
    cluster_id: int,
    cluster_labels: np.ndarray,
    X_pca: np.ndarray,
) -> str:
    mask = cluster_labels == cluster_id
    if not np.any(mask):
        raise ValueError(f"No maps found for cluster {cluster_id}")

    cluster_points = X_pca[mask]
    centroid = cluster_points.mean(axis=0)
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    sorted_idx = np.argsort(distances)

    cluster_df = df_labeled.loc[mask].reset_index(drop=True)
    top_n = min(3, len(cluster_df))
    candidates = cluster_df.iloc[sorted_idx[:top_n]]
    if "free_ratio" not in candidates.columns:
        return str(candidates.iloc[0]["map_name"])

    chosen = candidates.sort_values("free_ratio", ascending=False).iloc[0]
    return str(chosen["map_name"])


def _index_map_paths(data_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for path in data_dir.rglob("*.map"):
        paths[path.name] = path
    return paths


def _load_grid(map_path: Path) -> np.ndarray | None:
    try:
        return parse_map_file(map_path)
    except Exception as exc:
        print(f"[WARNING] Failed to parse {map_path.name}: {exc}")
        return None


def _train_cluster_worker(task: Dict[str, object]) -> Dict[str, object]:
    cluster_id = int(task["cluster_id"])
    map_name = str(task["map_name"])
    map_path = Path(task["map_path"])
    map_type = str(task.get("map_type", "unknown"))

    np.random.seed(config.RANDOM_STATE + cluster_id)

    grid = _load_grid(map_path)
    if grid is None:
        return {"cluster_id": cluster_id, "status": "failed", "map_name": map_name}

    env = GridEnvironment(grid)
    if len(env.free_cells) < 2:
        return {"cluster_id": cluster_id, "status": "skipped", "map_name": map_name}

    print(
        f"[INFO] Cluster {cluster_id}: grid={grid.shape} free_cells={len(env.free_cells)} "
        f"max_steps={env.max_steps}"
    )
    print(f"[INFO] Cluster {cluster_id}: training on {map_name}")
    print(f"[INFO] Cluster {cluster_id}: map type {map_type}")

    agent = QAgent(
        grid.shape,
        lr=config.LEARNING_RATE,
        gamma=config.DISCOUNT_FACTOR,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_min=config.EPSILON_MIN,
    )

    rewards, steps, _ = train_agent(
        env,
        agent,
        n_episodes=config.Q_LEARNING_EPISODES,
        log_every=config.RL_LOG_EVERY,
        step_log_interval=config.RL_STEP_LOG_INTERVAL,
        early_stop_window=config.RL_EARLY_STOP_WINDOW,
        early_stop_patience=config.RL_EARLY_STOP_PATIENCE,
        early_stop_min_delta=config.RL_EARLY_STOP_MIN_DELTA,
    )

    metrics = evaluate_agent(env, agent)
    q_path = config.MODELS_DIR / f"qtable_cluster_{cluster_id}.npy"
    try:
        np.save(q_path, agent.Q)
        print(f"[OK] Saved Q-table to {q_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save Q-table: {exc}")

    plot_training_curves(rewards, steps, cluster_id)

    return {
        "cluster_id": cluster_id,
        "status": "ok",
        "map_name": map_name,
        "metrics": metrics,
        "qtable_path": str(q_path),
    }


def train_cluster_agents(
    df_labeled: pd.DataFrame,
    cluster_labels: np.ndarray,
    X_pca: np.ndarray,
    data_dir: Path,
    config_module=config,
    parallel: bool = False,
) -> Dict[int, Dict[str, object]]:
    print("=== PHASE 6: Training cluster agents ===")
    data_dir = Path(data_dir)
    map_paths = _index_map_paths(data_dir)

    agents: Dict[int, Dict[str, object]] = {}
    tasks: List[Dict[str, object]] = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        try:
            map_name = select_representative_map(df_labeled, cluster_id, cluster_labels, X_pca)
        except ValueError as exc:
            print(f"[WARNING] {exc}")
            continue

        map_path = map_paths.get(map_name)
        if map_path is None:
            print(f"[WARNING] Map not found on disk: {map_name}")
            continue

        map_type = "unknown"
        if "label_map_type" in df_labeled.columns:
            row = df_labeled[df_labeled["map_name"] == map_name]
            if not row.empty:
                map_type = str(row.iloc[0]["label_map_type"])

        tasks.append(
            {
                "cluster_id": cluster_id,
                "map_name": map_name,
                "map_path": str(map_path),
                "map_type": map_type,
            }
        )

    if parallel and len(tasks) > 1:
        n_jobs = min(len(tasks), os.cpu_count() or 1)
        print(f"[INFO] Parallel training across {n_jobs} workers")
        with mp.Pool(processes=n_jobs) as pool:
            for result in pool.imap_unordered(_train_cluster_worker, tasks):
                if result.get("status") != "ok":
                    print(f"[WARNING] Cluster {result.get('cluster_id')} training skipped/failed")
                    continue
                cluster_id = int(result["cluster_id"])
                q_path = Path(result["qtable_path"])
                try:
                    q_table = np.load(q_path)
                    agent = QAgent(q_table.shape[:2])
                    agent.Q = q_table
                    agent.epsilon = 0.0
                except Exception as exc:
                    print(f"[WARNING] Failed to load Q-table for cluster {cluster_id}: {exc}")
                    continue

                agents[cluster_id] = {
                    "agent": agent,
                    "metrics": result.get("metrics", {}),
                    "map_name": result.get("map_name"),
                }
    else:
        for task in tasks:
            result = _train_cluster_worker(task)
            if result.get("status") != "ok":
                print(f"[WARNING] Cluster {result.get('cluster_id')} training skipped/failed")
                continue
            cluster_id = int(result["cluster_id"])
            q_path = Path(result["qtable_path"])
            try:
                q_table = np.load(q_path)
                agent = QAgent(q_table.shape[:2])
                agent.Q = q_table
                agent.epsilon = 0.0
            except Exception as exc:
                print(f"[WARNING] Failed to load Q-table for cluster {cluster_id}: {exc}")
                continue

            agents[cluster_id] = {
                "agent": agent,
                "metrics": result.get("metrics", {}),
                "map_name": result.get("map_name"),
            }

    return agents
