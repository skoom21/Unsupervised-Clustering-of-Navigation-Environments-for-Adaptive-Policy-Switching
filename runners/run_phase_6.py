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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from src.preprocessing import prepare_numeric_features, handle_missing_values
from src.regression import train_regression_models, evaluate_regression
from src.rl_agent import GridEnvironment, QAgent, downsample_grid, train_cluster_agents
from src.data_loader import parse_map_file

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


def _index_map_paths(data_dir: Path) -> dict:
    paths = {}
    for path in data_dir.rglob("*.map"):
        paths[path.name] = path
    return paths


def _run_greedy_episodes(env: GridEnvironment, agent: QAgent, n_episodes: int) -> dict:
    """Run greedy episodes for avg_steps_to_goal computation.

    Returns a dict with four metrics:
      - avg_steps          : mean steps over ALL episodes (censored at max_steps)
      - success_rate       : fraction of episodes that reached the goal
      - success_steps      : mean steps over SUCCESSFUL episodes only (uncensored)
      - failure_rate       : 1 - success_rate (explicit timeout fraction)

    NOTE: avg_steps_to_goal is an *auxiliary heuristic signal* — not an
    optimal plan length.  Unvisited states fall back to a Manhattan-greedy
    heuristic so the agent never silently loops on an all-zero Q-row.
    """
    steps_all = []
    steps_success = []
    successes = []
    for _ in range(n_episodes):
        try:
            state = env.reset()
        except ValueError:
            return {"avg_steps": float("nan"), "success_rate": 0.0,
                    "success_steps": float("nan"), "failure_rate": 1.0}
        done = False
        while not done:
            if agent.is_state_visited(state):
                action = agent.get_greedy_action(state)
            else:
                action = agent.heuristic_action(state, env.goal)
            state, _, done = env.step(action)
        reached = int(env.agent_pos == env.goal)
        steps_all.append(env.steps)
        successes.append(reached)
        if reached:
            steps_success.append(env.steps)

    success_rate = float(np.mean(successes))
    return {
        "avg_steps":    float(np.mean(steps_all)),
        "success_rate": success_rate,
        "success_steps": float(np.mean(steps_success)) if steps_success else float("nan"),
        "failure_rate": 1.0 - success_rate,
    }


def _compute_steps_to_goal(
    cluster_agents: dict,
    data_dir: Path,
    n_episodes: int = 100,
) -> pd.DataFrame:
    """Evaluate cluster agents only on their OWN training maps (representatives).
    This ensures we are measuring learned performance on the correct known environments.
    """
    total = len(cluster_agents)
    logger.info(f"\n=== PHASE 6: Evaluating cluster agents on training maps ({total} clusters, {n_episodes} episodes each) ===")
    data_dir = Path(data_dir)
    map_paths = _index_map_paths(data_dir)
    results = []

    for idx, (cluster_id, agent_info) in enumerate(cluster_agents.items()):
        progress_str = f"[{idx+1:3d}/{total}]"
        map_name = agent_info["map_name"]
        
        map_path = map_paths.get(map_name)
        if map_path is None:
            logger.info(f"{progress_str} SKIP  {map_name} — file not found")
            continue

        try:
            grid = parse_map_file(map_path)
        except Exception as exc:
            logger.info(f"{progress_str} SKIP  {map_name} — parse error: {exc}")
            continue

        orig_shape = grid.shape
        target_shape = agent_info.get("train_grid_shape")
        if target_shape:
            grid = downsample_grid(grid, max(target_shape))
        else:
            grid = downsample_grid(grid, config.RL_MAX_GRID_DIM)

        # Load the canonical goal used during training
        goal_path = config.MODELS_DIR / f"canonical_goal_cluster_{cluster_id}.npy"
        if goal_path.exists():
            canonical_goal = tuple(np.load(goal_path).tolist())
        else:
            canonical_goal = None
            logger.warning(f"No canonical goal found for cluster {cluster_id}, using random")

        env = GridEnvironment(grid, max_steps=config.RL_MAX_STEPS, canonical_goal=canonical_goal)
        agent = agent_info["agent"]
        epi = _run_greedy_episodes(env, agent, n_episodes)
        
        logger.info(
            f"{progress_str} CLUSTER {cluster_id} | {map_name:<30s} "
            f"success={epi['success_rate']:.2f}  "
            f"avg_steps={epi['avg_steps']:.1f}  "
            f"success_steps={epi['success_steps']:.1f}"
        )
        results.append({
            "map_name": map_name,
            "avg_steps_to_goal": epi["avg_steps"],
            "success_conditioned_steps": epi["success_steps"],
            "success_rate": epi["success_rate"],
            "failure_rate": epi["failure_rate"],
            "cluster_id": cluster_id,
        })

    df_out = pd.DataFrame(results)
    logger.info("\n--- avg_steps_to_goal summary per cluster ---")
    valid = df_out.dropna(subset=["success_rate"])
    if not valid.empty:
        summary = valid.groupby("cluster_id")[[
            "success_rate", "failure_rate", "avg_steps_to_goal", "success_conditioned_steps"
        ]].mean()
        logger.info(summary.to_string())
    logger.info(f"--- Total: {len(df_out)} maps, {df_out['avg_steps_to_goal'].isna().sum()} skipped ---\n")
    return df_out


def _save_steps_to_goal(df_steps: pd.DataFrame) -> None:
    output_path = config.PROCESSED_DATA_DIR / "steps_to_goal.csv"
    try:
        df_steps.to_csv(output_path, index=False)
        logger.info(f"Saved steps-to-goal to {output_path}")
    except OSError as exc:
        logger.warning(f"Failed to save steps-to-goal: {exc}")


def _save_phase6_scaler(scaler: StandardScaler) -> None:
    scaler_path = config.MODELS_DIR / "scaler_phase6_steps.pkl"
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved Phase 6 scaler to {scaler_path}")
    except OSError as exc:
        logger.warning(f"Failed to save Phase 6 scaler: {exc}")


def _run_steps_to_goal_regression(df_labeled: pd.DataFrame) -> None:
    X_numeric, _ = prepare_numeric_features(df_labeled)
    leakage_cols = [
        "avg_steps_to_goal",
        "avg_path_length",
        "max_path_length",
        "min_path_length",
        "n_reachable_pairs",
    ]
    X_numeric = X_numeric.drop(columns=[col for col in leakage_cols if col in X_numeric.columns])
    X_clean = handle_missing_values(X_numeric)

    # FIX 2: Prioritize BFS path length as the regression target.
    # RL heuristic is too noisy/biased for per-map regression.
    if "avg_path_length" in df_labeled.columns and df_labeled["avg_path_length"].notna().sum() > 10:
        target_col = "avg_path_length"
        target_name_suffix = "avg_path_length"
        logger.info("Using avg_path_length (BFS) as regression target")
    elif "success_conditioned_steps" in df_labeled.columns:
        target_col = "success_conditioned_steps"
        target_name_suffix = "success_steps"
        logger.warning("Falling back to RL success_conditioned_steps")
    else:
        raise ValueError("No valid regression target found (BFS data missing)")

    target = df_labeled[target_col]
    mask = target.notna()
    logger.info(f"Regression: {mask.sum()} valid samples (dropped {(~mask).sum()} nulls)")
    assert mask.sum() >= 50, f"Too few valid regression samples: {mask.sum()}"

    if not mask.any():
        raise ValueError("No valid avg_steps_to_goal values available for regression")

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
    _save_phase6_scaler(scaler)

    linear_model, poly_model = train_regression_models(
        X_train_scaled, y_train, target_name=target_name_suffix
    )

    metrics = {}
    metrics["linear"] = evaluate_regression(
        linear_model, X_test_scaled, y_test, f"linear_{target_name_suffix}"
    )
    metrics["polynomial"] = evaluate_regression(
        poly_model, X_test_scaled, y_test, f"poly_{target_name_suffix}"
    )

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.index.name = "model"
    report_path = config.REPORTS_DIR / f"regression_{target_name_suffix}_metrics.csv"
    try:
        metrics_df.to_csv(report_path)
        logger.info(f"Saved regression metrics to {report_path}")
    except OSError as exc:
        logger.warning(f"Failed to save regression metrics: {exc}")


def run_phase_6() -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6: Q-Learning")
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

    cluster_agents = train_cluster_agents(
        df_labeled,
        cluster_labels,
        X_pca,
        config.RAW_DATA_DIR,
        config,
        parallel=True,
    )

    metrics_rows = []
    reachability_rows = []
    for cluster_id, info in cluster_agents.items():
        metrics = info.get("metrics", {})
        metrics_rows.append(
            {
                "cluster_id": cluster_id,
                "map_name": info.get("map_name"),
                "success_rate": metrics.get("success_rate"),
                "mean_cumulative_reward": metrics.get("mean_cumulative_reward"),
                "mean_steps_to_goal": metrics.get("mean_steps_to_goal"),
            }
        )
        reachability_rows.append(
            {
                "cluster_id": cluster_id,
                "map_name": info.get("map_name"),
                "canonical_goal": info.get("canonical_goal"),
                "reachable_fraction": info.get("reachable_fraction"),
                "reachability_method": info.get("reachability_method"),
            }
        )

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        report_path = config.REPORTS_DIR / "rl_cluster_metrics.csv"
        try:
            metrics_df.to_csv(report_path, index=False)
            logger.info(f"Saved RL cluster metrics to {report_path}")
        except OSError as exc:
            logger.warning(f"Failed to save RL cluster metrics: {exc}")

    if reachability_rows:
        reach_df = pd.DataFrame(reachability_rows)
        reach_path = config.REPORTS_DIR / "reachability_analysis.csv"
        try:
            reach_df.to_csv(reach_path, index=False)
            logger.info(f"Saved reachability analysis to {reach_path}")
        except OSError as exc:
            logger.warning(f"Failed to save reachability analysis: {exc}")

    # FIX 2: Load path lengths from Phase 5 for regression ground truth.
    path_file = config.PROCESSED_DATA_DIR / "path_lengths.csv"
    if path_file.exists():
        df_paths = pd.read_csv(path_file)
        if "map_name" in df_paths.columns:
            df_labeled = df_labeled.merge(df_paths[["map_name", "avg_path_length"]], on="map_name", how="left")
            logger.info(f"Merged BFS path lengths for {df_labeled['avg_path_length'].notna().sum()} maps")

    df_steps = _compute_steps_to_goal(
        cluster_agents,
        config.RAW_DATA_DIR,
        n_episodes=100,
    )
    _save_steps_to_goal(df_steps)

    _run_steps_to_goal_regression(df_labeled)

    logger.info("Phase 6 Q-Learning complete")


def main() -> None:
    try:
        run_phase_6()
    except Exception as exc:
        logger.error(f"Phase 6 failed: {exc}")


if __name__ == "__main__":
    main()
