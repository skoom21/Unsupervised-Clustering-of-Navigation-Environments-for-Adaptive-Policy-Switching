import logging
import pandas as pd

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Tuple, Dict

import config

def compile_master_metrics(reports_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compiles all relevant output metrics into a single wide DataFrame and a single long DataFrame.
    """
    long_rows = []
    wide_rows = []

    def process_df(csv_name: str, task_name: str, default_model: str = "global"):
        p = reports_dir / csv_name
        if not p.exists():
            return
        
        df = pd.read_csv(p)
        df_wide = df.copy()
        df_wide["task"] = task_name
        
        # Some tables don't have a 'model' column. Insert it for the wide table uniformity if possible.
        if "model" not in df_wide.columns and default_model != "global":
             df_wide.insert(0, "model", default_model)

        wide_rows.append(df_wide)
        
        if "model" in df.columns:
            id_cols = ["model"]
        elif "map_name" in df.columns:
            id_cols = ["map_name"]
        elif "cluster_id" in df.columns:
            id_cols = ["cluster_id"]
        else:
            id_cols = []
            
        melt_id_cols = id_cols.copy()
        df_long = df.copy()
        df_long["task"] = task_name
        melt_id_cols.append("task")

        melted = df_long.melt(id_vars=melt_id_cols, var_name="metric", value_name="value")
        
        id_val_col = id_cols[0] if len(id_cols) > 0 else None
        
        for _, row in melted.iterrows():
            long_rows.append({
                "task": row["task"],
                "model_or_context": row[id_val_col] if id_val_col else default_model,
                "metric": row["metric"],
                "value": row["value"]
            })

    process_df("classification_map_type_metrics.csv", "classification_map_type")
    process_df("classification_difficulty_metrics.csv", "classification_difficulty")
    process_df("classification_density_category_metrics.csv", "classification_density")
    process_df("regression_avg_path_length_metrics.csv", "regression_path_length")
    process_df("regression_avg_steps_to_goal_metrics.csv", "regression_steps_to_goal")
    process_df("clustering_metrics.csv", "clustering", default_model="kmeans")
    process_df("rl_cluster_metrics.csv", "rl_cluster_training")
    process_df("adaptive_vs_single.csv", "rl_adaptive_inference")

    master_wide = pd.concat(wide_rows, ignore_index=True) if wide_rows else pd.DataFrame()
    master_long = pd.DataFrame(long_rows)

    return master_wide, master_long


def generate_comparison_tables(reports_dir: Path) -> Dict[str, str]:
    """
    Generates formatted Markdown tables requested by the benchmark reporting spec.
    """
    tables = {}
    
    # Table 1-3
    for title, fname in [
        ("Table 1 (Classification - map_type)", "classification_map_type_metrics.csv"),
        ("Table 2 (Classification - difficulty)", "classification_difficulty_metrics.csv"),
        ("Table 3 (Classification - density)", "classification_density_category_metrics.csv"),
    ]:
        if (reports_dir / fname).exists():
            df = pd.read_csv(reports_dir / fname)
            tables[title] = df.to_markdown(index=False)
            
    # Table 4 (Regression combined)
    df4a_path = reports_dir / "regression_avg_path_length_metrics.csv"
    df4b_path = reports_dir / "regression_avg_steps_to_goal_metrics.csv"
    
    reg_dfs = []
    if df4a_path.exists():
        d = pd.read_csv(df4a_path)
        d.insert(1, "target", "avg_path_length")
        reg_dfs.append(d)
    if df4b_path.exists():
        d = pd.read_csv(df4b_path)
        d.insert(1, "target", "avg_steps_to_goal")
        reg_dfs.append(d)
        
    if reg_dfs:
        df4 = pd.concat(reg_dfs, ignore_index=True)[["model", "target", "rmse", "mae", "r2"]]
        tables["Table 4 (Regression)"] = df4.to_markdown(index=False)

    # Table 5
    if (reports_dir / "clustering_metrics.csv").exists():
        df5 = pd.read_csv(reports_dir / "clustering_metrics.csv")
        tables["Table 5 (Clustering)"] = df5.to_markdown(index=False)

    # Table 6
    if (reports_dir / "rl_cluster_metrics.csv").exists():
        df6 = pd.read_csv(reports_dir / "rl_cluster_metrics.csv")
        df6 = df6[["cluster_id", "success_rate", "mean_cumulative_reward", "mean_steps_to_goal"]]
        tables["Table 6 (RL per cluster)"] = df6.to_markdown(index=False)

    # Table 7
    if (reports_dir / "adaptive_vs_single.csv").exists():
        df_adaptive = pd.read_csv(reports_dir / "adaptive_vs_single.csv")
        table7 = df_adaptive.groupby("map_type").agg(
            adaptive_success=("adaptive_success_rate", "mean"),
            single_success=("single_success_rate", "mean"),
            adaptive_steps=("adaptive_mean_steps", "mean"),
            single_steps=("single_mean_steps", "mean"),
            n_maps=("map_name", "count")
        ).reset_index()
        tables["Table 7 (Adaptive vs Single)"] = table7.to_markdown(index=False)

    return tables
