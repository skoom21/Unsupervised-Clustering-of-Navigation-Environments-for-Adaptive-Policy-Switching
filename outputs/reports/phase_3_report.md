# Phase 3 Output Report

Date: 2026-04-20

## Overview
- Completed Phase 3 clustering tasks (k selection, KMeans, evaluation, visualizations, and cross-tabs).
- Clustering was run on StandardScaler -> PCA transformed numeric features derived from the labeled dataset.

## K Selection Results
- Metrics evaluated for k=2..10 (inertia, silhouette, Davies-Bouldin).
- Recommended k (max silhouette): 2
- Summary (selected):
  - k=2: silhouette=0.5270, dbi=0.7470
  - k=4: silhouette=0.3808, dbi=1.0266
  - k=10: silhouette=0.4772, dbi=0.7503

## KMeans Training
- n_clusters used (downstream): 4
- Cluster sizes (k=4): {0: 100, 1: 30, 2: 80, 3: 50}
- Model saved.

## Clustering Evaluation
- Silhouette: 0.3808
- Davies-Bouldin: 1.0266
- Calinski-Harabasz: 165.8885

## Cross-Tabulation Highlights
- Cluster 1 is entirely street maps (30/30).
- Cluster 2 groups maze + room (40/40 split).
- Cluster 3 groups maze + random (20/30 split).
- Cluster 0 mixes random + street (40/60 split).

## K=2 Comparison Run
- Cluster sizes: {0: 230, 1: 30}
- Metrics: silhouette=0.5270, davies_bouldin=0.7470, calinski_harabasz=145.1711
- Cross-tab vs map type (counts):
  - Cluster 0: maze=60, room=40, random=70, street=60
  - Cluster 1: street=30
- Interpretation: k=2 maximizes silhouette but collapses most environments into a single mixed cluster.

## Outputs Generated
- Downstream (k=4, default filenames):
  - KMeans model: [outputs/models/kmeans_model.pkl](outputs/models/kmeans_model.pkl)
  - Clustering metrics: [outputs/reports/clustering_metrics.csv](outputs/reports/clustering_metrics.csv)
  - Cluster visualization (2D): [outputs/figures/cluster_visualization_2d.png](outputs/figures/cluster_visualization_2d.png)
  - Cluster profiles: [outputs/reports/cluster_profiles.csv](outputs/reports/cluster_profiles.csv)
  - Cluster profile plot: [outputs/figures/cluster_profiles.png](outputs/figures/cluster_profiles.png)
  - Cluster vs map type: [outputs/reports/cluster_vs_maptype.csv](outputs/reports/cluster_vs_maptype.csv)
  - Cluster vs difficulty: [outputs/reports/cluster_vs_difficulty.csv](outputs/reports/cluster_vs_difficulty.csv)
  - Cluster vs density: [outputs/reports/cluster_vs_density.csv](outputs/reports/cluster_vs_density.csv)
  - Cluster sample maps: [outputs/figures/cluster_sample_maps.png](outputs/figures/cluster_sample_maps.png)
  - Cluster labels: [outputs/models/cluster_labels_k4.npy](outputs/models/cluster_labels_k4.npy)
- Metric-optimal comparison (k=2, _k2 suffix):
  - KMeans model: [outputs/models/kmeans_model_k2.pkl](outputs/models/kmeans_model_k2.pkl)
  - Clustering metrics: [outputs/reports/clustering_metrics_k2.csv](outputs/reports/clustering_metrics_k2.csv)
  - Cluster visualization (2D): [outputs/figures/cluster_visualization_2d_k2.png](outputs/figures/cluster_visualization_2d_k2.png)
  - Cluster profiles: [outputs/reports/cluster_profiles_k2.csv](outputs/reports/cluster_profiles_k2.csv)
  - Cluster profile plot: [outputs/figures/cluster_profiles_k2.png](outputs/figures/cluster_profiles_k2.png)
  - Cluster vs map type: [outputs/reports/cluster_vs_maptype_k2.csv](outputs/reports/cluster_vs_maptype_k2.csv)
  - Cluster vs difficulty: [outputs/reports/cluster_vs_difficulty_k2.csv](outputs/reports/cluster_vs_difficulty_k2.csv)
  - Cluster vs density: [outputs/reports/cluster_vs_density_k2.csv](outputs/reports/cluster_vs_density_k2.csv)
  - Cluster sample maps: [outputs/figures/cluster_sample_maps_k2.png](outputs/figures/cluster_sample_maps_k2.png)
  - Cluster labels: [outputs/models/cluster_labels_k2.npy](outputs/models/cluster_labels_k2.npy)
- K selection metrics: [outputs/reports/k_selection_metrics.csv](outputs/reports/k_selection_metrics.csv)
- K selection plot: [outputs/figures/optimal_k_analysis.png](outputs/figures/optimal_k_analysis.png)

## Notes
- The silhouette-based optimum is k=2, which produces one small street-only cluster and one mixed cluster.
- The project prompt expects 4 clusters (matching map types). For downstream tasks (RL + policy switching), k=4 is selected for interpretability despite lower separation metrics.
- Decision rationale: k=2 maximizes silhouette but collapses most environments into one cluster; k=4 yields more meaningful structure aligned with navigation complexity.
- Street environments show intra-class variability (one cluster is purely street while others mix street with random), which should be noted in analysis.
- Clustering groups environments by structural navigability rather than explicit map type labels, indicating that feature representation captures underlying spatial complexity.
- Calinski-Harabasz index increases for k=4, suggesting improved cluster dispersion despite lower silhouette.
