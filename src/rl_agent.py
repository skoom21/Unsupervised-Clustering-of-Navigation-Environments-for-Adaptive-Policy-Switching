from pathlib import Path
from typing import Dict, List, Tuple
import os
import multiprocessing as mp

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

import config
from src.data_loader import parse_map_file

np.random.seed(42)


def find_canonical_goal(free_cells: list, grid_shape: tuple) -> tuple:
    """Return the free cell closest to the grid centre.

    Used to fix a consistent goal for both training and evaluation so
    the Q-table is judged on the same navigation task it was trained on.
    """
    if not free_cells:
        raise ValueError("No free cells available to select canonical goal")
    cr, cc = grid_shape[0] / 2.0, grid_shape[1] / 2.0
    return min(free_cells, key=lambda rc: (rc[0] - cr) ** 2 + (rc[1] - cc) ** 2)


class GridEnvironment:
    def __init__(
        self,
        grid: np.ndarray,
        max_steps: int | None = None,
        start_near_goal_prob: float = 0.0,
        near_goal_radius: int = 5,
        canonical_goal: tuple | None = None,
    ):
        self.grid = grid
        self.H, self.W = grid.shape
        default_max = self.H * self.W * 2
        if max_steps is None:
            self.max_steps = default_max
        else:
            self.max_steps = min(int(max_steps), default_max)
        self.start_near_goal_prob = max(0.0, min(1.0, float(start_near_goal_prob)))
        self.near_goal_radius = max(1, int(near_goal_radius))
        self.free_cells = [(r, c) for r in range(self.H) for c in range(self.W) if grid[r, c] == 0]
        self.canonical_goal = canonical_goal
        self.agent_pos = None
        self.goal = None
        self.steps = 0

    @staticmethod
    def _manhattan(a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _sample_non_goal(self, goal: tuple) -> tuple:
        candidates = [c for c in self.free_cells if c != goal]
        if not candidates:
            raise ValueError("Not enough free cells to sample start distinct from goal")
        return candidates[int(np.random.randint(len(candidates)))]

    def reset(self) -> tuple:
        if len(self.free_cells) < 2:
            raise ValueError("Not enough free cells to sample start/goal")

        if self.canonical_goal is not None:
            # Fixed goal — only randomize start
            self.goal = self.canonical_goal
            self.agent_pos = self._sample_non_goal(self.goal)
        else:
            # Random goal — multi-goal training
            indices = np.random.choice(len(self.free_cells), size=2, replace=False)
            self.agent_pos = self.free_cells[indices[0]]
            self.goal = self.free_cells[indices[1]]

        if np.random.rand() < self.start_near_goal_prob:
            radius = self.near_goal_radius
            near_cells = [
                cell
                for cell in self.free_cells
                if cell != self.goal and self._manhattan(cell, self.goal) <= radius
            ]
            if near_cells:
                self.agent_pos = near_cells[int(np.random.randint(len(near_cells)))]
        
        self.steps = 0
        return self.agent_pos

    def step(self, action: int) -> tuple:
        prev_distance = self._manhattan(self.agent_pos, self.goal)
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        self.steps += 1
        moved = False
        if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] == 0:
            self.agent_pos = (nr, nc)
            reward = config.RL_REWARD_STEP
            moved = True
        else:
            reward = config.RL_REWARD_INVALID
        new_distance = self._manhattan(self.agent_pos, self.goal)
        if moved:
            reward += config.RL_REWARD_DISTANCE_SCALE * (prev_distance - new_distance)
        done = (self.agent_pos == self.goal) or (self.steps >= self.max_steps)
        if self.agent_pos == self.goal:
            reward = config.RL_REWARD_GOAL
        elif self.steps >= self.max_steps:
            reward += config.RL_REWARD_TIMEOUT
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
        return self.get_greedy_action(state)

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.lr * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def is_state_visited(self, state: tuple) -> bool:
        """Returns True if this state has ever been updated (Q-row != all zeros)."""
        try:
            return bool(np.any(self.Q[state] != 0.0))
        except (IndexError, TypeError):
            return False

    def get_greedy_action(self, state: tuple) -> int:
        """Return best Q-action; falls back to random if state is unvisited."""
        try:
            q_row = self.Q[state]
        except (IndexError, TypeError):
            return int(np.random.randint(4))
        if not np.any(q_row != 0.0):
            # Unvisited state — random is safer than argmax(all-zeros)
            return int(np.random.randint(4))
        return int(np.argmax(q_row))

    def heuristic_action(self, state: tuple, goal: tuple) -> int:
        """Manhattan-greedy: move in whichever cardinal direction reduces distance most."""
        r, c = state
        gr, gc = goal
        dr = gr - r
        dc = gc - c
        # Prefer the axis with the larger gap; break ties randomly
        if abs(dr) > abs(dc):
            return 0 if dr < 0 else 1   # up / down
        elif abs(dc) > abs(dr):
            return 2 if dc < 0 else 3   # left / right
        else:
            return int(np.random.choice([0 if dr < 0 else 1, 2 if dc < 0 else 3]))


def _reachability_fraction(grid: np.ndarray, goal: tuple, sample_size: int = 50) -> float:
    from collections import deque
    free_cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] == 0]
    if len(free_cells) < 2:
        return 0.0
    
    # Use a fixed seed for sampling to ensure deterministic checks
    rng = np.random.default_rng(42)
    indices = rng.choice(len(free_cells), size=min(sample_size, len(free_cells)), replace=False)
    samples = [free_cells[i] for i in indices]
    
    reachable = 0
    for start in samples:
        if start == goal:
            reachable += 1
            continue
        
        q = deque([start])
        visited = {start}
        found = False
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                found = True
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] \
                   and grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        if found:
            reachable += 1
    return reachable / len(samples)


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
    logger.info(f"[CONFIG] episodes={n_episodes} decay={config.EPSILON_DECAY} patience={early_stop_patience}")
    rewards: List[float] = []
    steps: List[int] = []
    success: List[int] = []
    best_avg = float("-inf")
    no_improve = 0

    p1 = float(getattr(config, "RL_CURRICULUM_P1", 0.0))
    p2 = float(getattr(config, "RL_CURRICULUM_P2", 0.0))
    prob_early = float(getattr(config, "RL_CURRICULUM_PROB_EARLY", 0.0))
    prob_mid = float(getattr(config, "RL_CURRICULUM_PROB_MID", 0.0))
    prob_late = float(getattr(config, "RL_CURRICULUM_PROB_LATE", 0.0))
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    if p2 < p1:
        p1, p2 = p2, p1

    for episode in range(n_episodes):
        if n_episodes > 1:
            progress = episode / float(n_episodes - 1)
        else:
            progress = 1.0

        if progress < p1:
            env.start_near_goal_prob = prob_early
        elif progress < p2:
            env.start_near_goal_prob = prob_mid
        else:
            env.start_near_goal_prob = prob_late
        try:
            state = env.reset()
        except ValueError as exc:
            logger.warning(f"Episode skipped: {exc}")
            break

        total_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        steps.append(env.steps)
        success.append(int(env.agent_pos == env.goal))

        if (episode + 1) % log_every == 0:
            window = min(log_every, len(rewards))
            avg_reward = float(np.mean(rewards[-window:]))
            avg_steps = float(np.mean(steps[-window:]))
            success_rate = float(np.mean(success[-window:]))
            logger.info(
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
                        logger.info(
                            f"[TRAIN] Early stopping at episode {episode + 1}: "
                            f"recent_avg={recent_avg:.2f} best_avg={best_avg:.2f}"
                        )
                        break

        agent.decay_epsilon()

    return rewards, steps, success


def evaluate_agent(env: GridEnvironment, agent: QAgent, n_eval: int = 100) -> Dict[str, float]:
    original_epsilon = agent.epsilon
    original_start_prob = getattr(env, "start_near_goal_prob", 0.0)
    agent.epsilon = 0.0
    env.start_near_goal_prob = 0.0

    rewards = []
    steps = []
    successes = []
    try:
        for _ in range(n_eval):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                # Use heuristic fallback for unvisited states
                if agent.is_state_visited(state):
                    action = agent.get_greedy_action(state)
                else:
                    action = agent.heuristic_action(state, env.goal)
                state, reward, done = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
            steps.append(env.steps)
            successes.append(int(env.agent_pos == env.goal))
    except ValueError as exc:
        logger.warning(f"Evaluation skipped: {exc}")
    finally:
        agent.epsilon = original_epsilon
        env.start_near_goal_prob = original_start_prob

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
        logger.warning(f"No rewards to plot for cluster {cluster_id}")
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
        logger.info(f"Saved training curve to {fig_path}")
    except OSError as exc:
        logger.warning(f"Failed to save training curve: {exc}")
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

    cluster_indices = np.where(mask)[0]
    cluster_points = X_pca[mask]
    cluster_df = df_labeled.iloc[cluster_indices].reset_index(drop=True)
    candidate_mask = np.ones(len(cluster_df), dtype=bool)

    if "height" in cluster_df.columns and "width" in cluster_df.columns:
        max_dim = getattr(config, "RL_MAX_TRAIN_MAP_DIM", None)
        if max_dim:
            candidate_mask = (cluster_df["height"] <= max_dim) & (cluster_df["width"] <= max_dim)
            if not candidate_mask.any():
                candidate_mask = np.ones(len(cluster_df), dtype=bool)

    candidate_points = cluster_points[candidate_mask]
    centroid = candidate_points.mean(axis=0)
    distances = np.linalg.norm(candidate_points - centroid, axis=1)
    sorted_idx = np.argsort(distances)

    candidate_df = cluster_df[candidate_mask].reset_index(drop=True)
    top_n = min(3, len(candidate_df))
    candidates = candidate_df.iloc[sorted_idx[:top_n]]
    # FIX 3: Representative map selection.
    # Prioritizes maps with high free_ratio and largest_free_component.
    if "free_ratio" not in candidates.columns or "largest_free_component" not in candidates.columns:
        return str(candidates.iloc[0]["map_name"])

    # Score based on navigability
    candidates = candidates.copy()
    candidates["nav_score"] = candidates["free_ratio"] * candidates["largest_free_component"]
    chosen = candidates.sort_values("nav_score", ascending=False).iloc[0]
    return str(chosen["map_name"])


def downsample_grid_adaptive(grid: np.ndarray, map_row: pd.Series | dict) -> np.ndarray:
    """FIX 1: Adaptive downsampling based on map specs.
    - 512x512 -> 64x64 (stride 8)
    - 1024x1024 -> 64x64 (stride 16)
    - 256x256 -> 256x256 (no downsampling)
    - Fine structures (low corridor width or few rooms) -> 128x128
    """
    h, w = grid.shape
    
    corridor_width = map_row.get("corridor_width", 100)
    num_rooms = map_row.get("num_rooms", 100)
    
    if corridor_width == 1 or (0 < num_rooms <= 16):
        target_dim = config.RL_GRID_DIM_FINE  # 128
    elif h >= 1024:
        target_dim = config.RL_GRID_DIM_1024  # 64
    elif h >= 512:
        target_dim = config.RL_GRID_DIM_512   # 64
    elif h <= 256:
        return grid
    else:
        target_dim = config.RL_MAX_GRID_DIM    # fallback 64

    stride = int(np.ceil(max(h, w) / target_dim))
    stride = max(1, stride)
    return grid[::stride, ::stride]


def downsample_grid(grid: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None:
        return grid
    h, w = grid.shape
    stride = int(np.ceil(max(h, w) / max_dim))
    stride = max(1, stride)
    return grid[::stride, ::stride]


def _index_map_paths(data_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for path in data_dir.rglob("*.map"):
        paths[path.name] = path
    return paths


def _load_grid(map_path: Path) -> np.ndarray | None:
    try:
        return parse_map_file(map_path)
    except Exception as exc:
        logger.warning(f"Failed to parse {map_path.name}: {exc}")
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

    # FIX: Restore downsampling before computing canonical goal
    row = task.get("map_features", {})
    grid = downsample_grid_adaptive(grid, row)

    # Compute canonical goal on the DOWNSAMPLED grid
    free_cells_all = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] == 0]
    canonical_goal = find_canonical_goal(free_cells_all, grid.shape)
    
    # Reachability sanity check
    frac = _reachability_fraction(grid, canonical_goal)
    logger.info(f"Cluster {cluster_id}: downsampled to {grid.shape}, canonical_goal={canonical_goal} reachable_from={frac:.0%} of sampled starts")
    if frac < 0.5:
        logger.warning(f"Cluster {cluster_id}: canonical goal is poorly connected — consider a different representative map")

    env = GridEnvironment(
        grid,
        max_steps=config.RL_MAX_STEPS,
        start_near_goal_prob=0.0, # Force long-range navigation learning
        near_goal_radius=config.RL_NEAR_GOAL_RADIUS,
        canonical_goal=canonical_goal,
    )
    if len(env.free_cells) < 2:
        return {"cluster_id": cluster_id, "status": "skipped", "map_name": map_name}

    logger.info(
        f"[INFO] Cluster {cluster_id}: grid={grid.shape} free_cells={len(env.free_cells)} "
        f"max_steps={env.max_steps}"
    )
    logger.info(f"Cluster {cluster_id}: training on {map_name}")
    logger.info(f"Cluster {cluster_id}: map type {map_type}")

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

    if agent.epsilon > 0.05:
        logger.info(
            f"[WARNING] Cluster {cluster_id} did not converge: "
            f"epsilon={agent.epsilon:.4f} after {config.Q_LEARNING_EPISODES} episodes. "
            f"Q-table will still be saved but results may be unreliable."
        )
    else:
        logger.info(f"Cluster {cluster_id} converged: epsilon={agent.epsilon:.4f}")

    metrics = evaluate_agent(env, agent)
    q_path = config.MODELS_DIR / f"qtable_cluster_{cluster_id}.npy"
    goal_path = config.MODELS_DIR / f"canonical_goal_cluster_{cluster_id}.npy"
    try:
        np.save(q_path, agent.Q)
        np.save(goal_path, np.array(canonical_goal))
        logger.info(f"Saved Q-table and goal to {config.MODELS_DIR}")
    except OSError as exc:
        logger.warning(f"Failed to save assets: {exc}")

    plot_training_curves(rewards, steps, cluster_id)

    return {
        "cluster_id": cluster_id,
        "status": "ok",
        "map_name": map_name,
        "metrics": metrics,
        "qtable_path": str(q_path),
        "goal_path": str(goal_path),
        "canonical_goal": canonical_goal,
        "train_grid_shape": grid.shape,
        "reachable_fraction": frac,
        "reachability_method": "BFS from 50 sampled free cells to canonical goal"
    }


def train_cluster_agents(
    df_labeled: pd.DataFrame,
    cluster_labels: np.ndarray,
    X_pca: np.ndarray,
    data_dir: Path,
    config_module=config,
    parallel: bool = False,
) -> Dict[int, Dict[str, object]]:
    logger.info("=== PHASE 6: Training cluster agents ===")
    data_dir = Path(data_dir)
    map_paths = _index_map_paths(data_dir)

    agents: Dict[int, Dict[str, object]] = {}
    tasks: List[Dict[str, object]] = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        try:
            map_name = select_representative_map(df_labeled, cluster_id, cluster_labels, X_pca)
        except ValueError as exc:
            logger.warning(f"{exc}")
            continue

        map_path = map_paths.get(map_name)
        if map_path is None:
            logger.warning(f"Map not found on disk: {map_name}")
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
                "map_features": df_labeled[df_labeled["map_name"] == map_name].iloc[0].to_dict(),
            }
        )

    if parallel and len(tasks) > 1:
        n_jobs = min(len(tasks), os.cpu_count() or 1)
        logger.info(f"Parallel training across {n_jobs} workers")
        with mp.Pool(processes=n_jobs) as pool:
            for result in pool.imap_unordered(_train_cluster_worker, tasks):
                if result.get("status") != "ok":
                    logger.warning(f"Cluster {result.get('cluster_id')} training skipped/failed")
                    continue
                cluster_id = int(result["cluster_id"])
                q_path = Path(result["qtable_path"])
                try:
                    q_table = np.load(q_path)
                    agent = QAgent(q_table.shape[:2])
                    agent.Q = q_table
                    agent.epsilon = 0.0
                except Exception as exc:
                    logger.warning(f"Failed to load Q-table for cluster {cluster_id}: {exc}")
                    continue

                agents[cluster_id] = {
                    "agent": agent,
                    "metrics": result.get("metrics", {}),
                    "map_name": result.get("map_name"),
                    "train_grid_shape": result.get("train_grid_shape"),
                    "canonical_goal": result.get("canonical_goal"),
                    "reachable_fraction": result.get("reachable_fraction"),
                    "reachability_method": result.get("reachability_method"),
                }
    else:
        for task in tasks:
            result = _train_cluster_worker(task)
            if result.get("status") != "ok":
                logger.warning(f"Cluster {result.get('cluster_id')} training skipped/failed")
                continue
            cluster_id = int(result["cluster_id"])
            q_path = Path(result["qtable_path"])
            try:
                q_table = np.load(q_path)
                agent = QAgent(q_table.shape[:2])
                agent.Q = q_table
                agent.epsilon = 0.0
            except Exception as exc:
                logger.warning(f"Failed to load Q-table for cluster {cluster_id}: {exc}")
                continue

            agents[cluster_id] = {
                "agent": agent,
                "metrics": result.get("metrics", {}),
                "map_name": result.get("map_name"),
                "train_grid_shape": result.get("train_grid_shape"),
                "canonical_goal": result.get("canonical_goal"),
                "reachable_fraction": result.get("reachable_fraction"),
                "reachability_method": result.get("reachability_method"),
            }

    return agents
