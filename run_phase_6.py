from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from src.preprocessing import prepare_numeric_features, handle_missing_values
from src.regression import train_regression_models, evaluate_regression
from src.rl_agent import GridEnvironment, QAgent, train_cluster_agents
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


def _index_map_paths(data_dir: Path) -> dict:
    paths = {}
    for path in data_dir.rglob("*.map"):
        paths[path.name] = path
    return paths


def _run_greedy_episodes(env: GridEnvironment, agent: QAgent, n_episodes: int) -> tuple:
    steps = []
    successes = []
    for _ in range(n_episodes):
        try:
            state = env.reset()
        except ValueError:
            return float("nan"), 0.0
        done = False
        while not done:
            action = agent.get_greedy_action(state)
            state, _, done = env.step(action)
        steps.append(env.steps)
        successes.append(int(env.agent_pos == env.goal))
    return float(np.mean(steps)), float(np.mean(successes))


def _compute_steps_to_goal(
    df_labeled: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_agents: dict,
    data_dir: Path,
    n_episodes: int = 50,
) -> pd.DataFrame:
    print("=== PHASE 6: Computing avg_steps_to_goal ===")
    data_dir = Path(data_dir)
    map_paths = _index_map_paths(data_dir)
    results = []

    for idx, map_name in enumerate(df_labeled["map_name"].tolist()):
        cluster_id = int(cluster_labels[idx])
        agent_info = cluster_agents.get(cluster_id)
        if agent_info is None:
            results.append(
                {
                    "map_name": map_name,
                    "avg_steps_to_goal": np.nan,
                    "success_rate": np.nan,
                    "cluster_id": cluster_id,
                }
            )
            continue

        map_path = map_paths.get(map_name)
        if map_path is None:
            print(f"[WARNING] Map not found on disk: {map_name}")
            results.append(
                {
                    "map_name": map_name,
                    "avg_steps_to_goal": np.nan,
                    "success_rate": np.nan,
                    "cluster_id": cluster_id,
                }
            )
            continue

        try:
            grid = parse_map_file(map_path)
        except Exception as exc:
            print(f"[WARNING] Failed to parse {map_name}: {exc}")
            results.append(
                {
                    "map_name": map_name,
                    "avg_steps_to_goal": np.nan,
                    "success_rate": np.nan,
                    "cluster_id": cluster_id,
                }
            )
            continue

        if grid.shape[0] >= 1024 or grid.shape[1] >= 1024:
            grid = grid[::4, ::4]

        env = GridEnvironment(grid)
        agent = agent_info["agent"]
        avg_steps, success_rate = _run_greedy_episodes(env, agent, n_episodes)

        results.append(
            {
                "map_name": map_name,
                "avg_steps_to_goal": avg_steps,
                "success_rate": success_rate,
                "cluster_id": cluster_id,
            }
        )

    return pd.DataFrame(results)


def _save_steps_to_goal(df_steps: pd.DataFrame) -> None:
    output_path = config.PROCESSED_DATA_DIR / "steps_to_goal.csv"
    try:
        df_steps.to_csv(output_path, index=False)
        print(f"[OK] Saved steps-to-goal to {output_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save steps-to-goal: {exc}")


def _save_phase6_scaler(scaler: StandardScaler) -> None:
    scaler_path = config.MODELS_DIR / "scaler_phase6_steps.pkl"
    try:
        joblib.dump(scaler, scaler_path)
        print(f"[OK] Saved Phase 6 scaler to {scaler_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save Phase 6 scaler: {exc}")


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

    if "avg_steps_to_goal" not in df_labeled.columns:
        raise ValueError("avg_steps_to_goal target not found; RL evaluation failed")

    target = df_labeled["avg_steps_to_goal"]
    mask = target.notna()
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
        X_train_scaled, y_train, target_name="avg_steps_to_goal"
    )

    metrics = {}
    metrics["linear"] = evaluate_regression(
        linear_model, X_test_scaled, y_test, "linear_avg_steps_to_goal"
    )
    metrics["polynomial"] = evaluate_regression(
        poly_model, X_test_scaled, y_test, "poly_avg_steps_to_goal"
    )

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.index.name = "model"
    report_path = config.REPORTS_DIR / "regression_avg_steps_to_goal_metrics.csv"
    try:
        metrics_df.to_csv(report_path)
        print(f"[OK] Saved regression metrics to {report_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save regression metrics: {exc}")


def run_phase_6() -> None:
    print("\n" + "=" * 60)
    print("PHASE 6: Q-Learning")
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

    cluster_agents = train_cluster_agents(
        df_labeled,
        cluster_labels,
        X_pca,
        config.RAW_DATA_DIR,
        config,
        parallel=True,
    )

    metrics_rows = []
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

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        report_path = config.REPORTS_DIR / "rl_cluster_metrics.csv"
        try:
            metrics_df.to_csv(report_path, index=False)
            print(f"[OK] Saved RL cluster metrics to {report_path}")
        except OSError as exc:
            print(f"[WARNING] Failed to save RL cluster metrics: {exc}")

    df_steps = _compute_steps_to_goal(
        df_labeled,
        cluster_labels,
        cluster_agents,
        config.RAW_DATA_DIR,
        n_episodes=50,
    )
    _save_steps_to_goal(df_steps)

    df_with_steps = df_labeled.merge(df_steps[["map_name", "avg_steps_to_goal"]], on="map_name", how="left")
    _run_steps_to_goal_regression(df_with_steps)

    print("[OK] Phase 6 Q-Learning complete")


def main() -> None:
    try:
        run_phase_6()
    except Exception as exc:
        print(f"[ERROR] Phase 6 failed: {exc}")


if __name__ == "__main__":
    main()
