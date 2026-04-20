from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

import config
from src.data_loader import parse_map_file

np.random.seed(42)


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)) -> Tuple[pd.DataFrame, int]:
    print("=== PHASE 3: K selection ===")
    results = []
    for k in k_range:
        try:
            model = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=config.RANDOM_STATE)
            labels = model.fit_predict(X_scaled)
            inertia = float(model.inertia_)
            sil = float(silhouette_score(X_scaled, labels))
            dbi = float(davies_bouldin_score(X_scaled, labels))
            results.append({"k": k, "inertia": inertia, "silhouette": sil, "davies_bouldin": dbi})
            print(f"k={k} inertia={inertia:.4f} silhouette={sil:.4f} dbi={dbi:.4f}")
        except Exception as exc:
            print(f"[WARNING] Failed metrics for k={k}: {exc}")

    df = pd.DataFrame(results)
    output_path = config.REPORTS_DIR / "k_selection_metrics.csv"
    try:
        df.to_csv(output_path, index=False)
        print(f"[OK] Saved k-selection metrics to {output_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save k-selection metrics: {exc}")

    if not df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].plot(df["k"], df["inertia"], marker="o")
        axes[0].set_title("Elbow (Inertia)")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Inertia")

        axes[1].plot(df["k"], df["silhouette"], marker="o")
        axes[1].set_title("Silhouette")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Score")

        axes[2].plot(df["k"], df["davies_bouldin"], marker="o")
        axes[2].set_title("Davies-Bouldin")
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("Score")

        plt.tight_layout()
        fig_path = config.FIGURES_DIR / "optimal_k_analysis.png"
        try:
            fig.savefig(fig_path, dpi=200)
            print(f"[OK] Saved k-selection plot to {fig_path}")
        except OSError as exc:
            print(f"[WARNING] Failed to save k-selection plot: {exc}")
        finally:
            plt.close(fig)

    recommended = int(df.loc[df["silhouette"].idxmax(), "k"]) if not df.empty else config.N_CLUSTERS
    print(f"Recommended k: {recommended}")
    return df, recommended


def add_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def run_kmeans(X_scaled: np.ndarray, n_clusters: int = 4, suffix: str = ""):
    print("=== PHASE 3: KMeans training ===")
    model = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=config.RANDOM_STATE)
    labels = model.fit_predict(X_scaled)

    counts = pd.Series(labels).value_counts().to_dict()
    print(f"Cluster sizes: {counts}")

    model_path = add_suffix(config.MODELS_DIR / "kmeans_model.pkl", suffix)
    try:
        joblib.dump(model, model_path)
        print(f"[OK] Saved KMeans model to {model_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save KMeans model: {exc}")

    return model, labels


def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray, suffix: str = "") -> Dict[str, float]:
    print("=== PHASE 3: Clustering evaluation ===")
    metrics = {}
    try:
        metrics["silhouette"] = float(silhouette_score(X_scaled, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_scaled, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, labels))
        print(f"Metrics: {metrics}")
    except Exception as exc:
        print(f"[WARNING] Failed to compute clustering metrics: {exc}")

    output_path = add_suffix(config.REPORTS_DIR / "clustering_metrics.csv", suffix)
    try:
        pd.DataFrame([metrics]).to_csv(output_path, index=False)
        print(f"[OK] Saved clustering metrics to {output_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save clustering metrics: {exc}")

    return metrics


def visualize_clusters_2d(X_2d: np.ndarray, labels: np.ndarray, suffix: str = "") -> None:
    print("=== PHASE 3: Cluster visualization (2D) ===")
    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {cluster_id}", alpha=0.7)
        centroid = X_2d[mask].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], color="black", marker="x", s=80)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()

    output_path = add_suffix(config.FIGURES_DIR / "cluster_visualization_2d.png", suffix)
    try:
        fig.savefig(output_path, dpi=200)
        print(f"[OK] Saved cluster visualization to {output_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save cluster visualization: {exc}")
    finally:
        plt.close(fig)


def cross_tabulate_clusters_vs_labels(
    df_labeled: pd.DataFrame,
    cluster_labels: np.ndarray,
    suffix: str = "",
) -> None:
    print("=== PHASE 3: Cluster vs label cross-tabs ===")
    df = df_labeled.copy()
    df["cluster_label"] = cluster_labels

    tables = {
        "maptype": "label_map_type",
        "difficulty": "label_difficulty",
        "density": "label_density_category",
    }

    for name, col in tables.items():
        if col not in df.columns:
            print(f"[WARNING] Missing label column: {col}")
            continue
        ctab = pd.crosstab(df["cluster_label"], df[col])
        pct = ctab.div(ctab.sum(axis=1), axis=0) * 100.0
        print(f"\nCluster vs {col} (counts):\n{ctab}")
        print(f"Cluster vs {col} (row %):\n{pct.round(2)}")

        output_path = add_suffix(config.REPORTS_DIR / f"cluster_vs_{name}.csv", suffix)
        try:
            ctab.to_csv(output_path)
            print(f"[OK] Saved {name} cross-tab to {output_path}")
        except OSError as exc:
            print(f"[WARNING] Failed to save {name} cross-tab: {exc}")


def analyze_cluster_profiles(df_features: pd.DataFrame, labels: np.ndarray, suffix: str = "") -> None:
    print("=== PHASE 3: Cluster profiles ===")
    df = df_features.copy()
    df["cluster_label"] = labels

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "cluster_label" in numeric_cols:
        numeric_cols.remove("cluster_label")

    means = df.groupby("cluster_label")[numeric_cols].mean()
    stds = df.groupby("cluster_label")[numeric_cols].std()
    profile = pd.concat(
        {"mean": means, "std": stds}, axis=1
    )

    output_path = add_suffix(config.REPORTS_DIR / "cluster_profiles.csv", suffix)
    try:
        profile.to_csv(output_path)
        print(f"[OK] Saved cluster profiles to {output_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save cluster profiles: {exc}")

    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        means.T.plot(kind="bar", ax=ax)
        ax.set_ylabel("Feature mean")
        ax.set_title("Cluster Feature Means")
        ax.legend(title="Cluster")
        plt.tight_layout()
        fig_path = add_suffix(config.FIGURES_DIR / "cluster_profiles.png", suffix)
        fig.savefig(fig_path, dpi=200)
        print(f"[OK] Saved cluster profile plot to {fig_path}")
    except Exception as exc:
        print(f"[WARNING] Failed to plot cluster profiles: {exc}")
    finally:
        plt.close("all")


def _find_map_path(data_dir: Path, map_name: str) -> Path:
    matches = list(data_dir.rglob(map_name))
    return matches[0] if matches else None


def visualize_cluster_samples(
    data_dir: Path,
    df_labeled: pd.DataFrame,
    cluster_labels: np.ndarray,
    suffix: str = "",
) -> None:
    print("=== PHASE 3: Cluster sample maps ===")
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[WARNING] Data directory does not exist: {data_dir}")
        return

    df = df_labeled.copy()
    df["cluster_label"] = cluster_labels
    clusters = sorted(df["cluster_label"].unique())

    n_samples = 3
    fig, axes = plt.subplots(len(clusters), n_samples, figsize=(n_samples * 3, len(clusters) * 3))
    if len(clusters) == 1:
        axes = np.array([axes])

    for row, cluster_id in enumerate(clusters):
        subset = df[df["cluster_label"] == cluster_id]
        sample_names = subset["map_name"].sample(
            n=min(n_samples, len(subset)),
            random_state=config.RANDOM_STATE,
        )
        for col in range(n_samples):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(sample_names):
                continue
            map_name = sample_names.iloc[col]
            map_path = _find_map_path(data_dir, map_name)
            if map_path is None:
                ax.set_title(f"Missing: {map_name}")
                continue
            try:
                grid = parse_map_file(map_path)
                ax.imshow(grid, cmap="binary")
                label = subset[subset["map_name"] == map_name]["label_map_type"].iloc[0]
                ax.set_title(f"C{cluster_id} | {map_name}\n{label}")
            except Exception as exc:
                ax.set_title(f"Error: {map_name}")
                print(f"[WARNING] Failed to load {map_name}: {exc}")
        axes[row, 0].set_ylabel(f"Cluster {cluster_id}")

    plt.tight_layout()
    output_path = add_suffix(config.FIGURES_DIR / "cluster_sample_maps.png", suffix)
    try:
        fig.savefig(output_path, dpi=200)
        print(f"[OK] Saved cluster sample maps to {output_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save cluster sample maps: {exc}")
    finally:
        plt.close(fig)
