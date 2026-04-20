# Phase 4 Output Report

Date: 2026-04-20

## Overview
- Completed Phase 4 classification for map type, difficulty, and density category.
- Metadata columns were excluded and a Phase 4 scaler was fitted on grid-derived features only.
- Leakage checks: no duplicate `map_name`, no train/test overlap, permutation sanity check near chance.

## Map Type Classification (4-class)
- KNN: accuracy=0.9615, precision=0.9663, recall=0.9615, f1=0.9617
- SVM: accuracy=0.9615, precision=0.9663, recall=0.9615, f1=0.9617
- Decision Tree: accuracy=1.0000, precision=1.0000, recall=1.0000, f1=1.0000
- Bagging: accuracy=1.0000, precision=1.0000, recall=1.0000, f1=1.0000
- AdaBoost: accuracy=0.9038, precision=0.9174, recall=0.9038, f1=0.9059

## Difficulty Classification (3-class)
- KNN: accuracy=0.9615, precision=0.9654, recall=0.9615, f1=0.9613
- SVM: accuracy=0.9615, precision=0.9654, recall=0.9615, f1=0.9613
- Decision Tree: accuracy=1.0000, precision=1.0000, recall=1.0000, f1=1.0000

## Density Category Classification (3-class)
- KNN: accuracy=1.0000, precision=1.0000, recall=1.0000, f1=1.0000
- SVM: accuracy=1.0000, precision=1.0000, recall=1.0000, f1=1.0000
- Decision Tree: accuracy=1.0000, precision=1.0000, recall=1.0000, f1=1.0000

## Outputs Generated
- Reports:
  - [outputs/reports/classifier_comparison.csv](outputs/reports/classifier_comparison.csv)
  - [outputs/reports/classification_map_type_metrics.csv](outputs/reports/classification_map_type_metrics.csv)
  - [outputs/reports/classification_difficulty_metrics.csv](outputs/reports/classification_difficulty_metrics.csv)
  - [outputs/reports/classification_density_category_metrics.csv](outputs/reports/classification_density_category_metrics.csv)
  - [outputs/reports/map_type_feature_importances.csv](outputs/reports/map_type_feature_importances.csv)
- Figures:
  - [outputs/figures/classifier_comparison.png](outputs/figures/classifier_comparison.png)
  - [outputs/figures/cm_knn_map_type.png](outputs/figures/cm_knn_map_type.png)
  - [outputs/figures/cm_svm_map_type.png](outputs/figures/cm_svm_map_type.png)
  - [outputs/figures/cm_dt_map_type.png](outputs/figures/cm_dt_map_type.png)
  - [outputs/figures/cm_bagging_map_type.png](outputs/figures/cm_bagging_map_type.png)
  - [outputs/figures/cm_boosting_map_type.png](outputs/figures/cm_boosting_map_type.png)
  - [outputs/figures/cm_knn_difficulty.png](outputs/figures/cm_knn_difficulty.png)
  - [outputs/figures/cm_svm_difficulty.png](outputs/figures/cm_svm_difficulty.png)
  - [outputs/figures/cm_dt_difficulty.png](outputs/figures/cm_dt_difficulty.png)
  - [outputs/figures/cm_knn_density_category.png](outputs/figures/cm_knn_density_category.png)
  - [outputs/figures/cm_svm_density_category.png](outputs/figures/cm_svm_density_category.png)
  - [outputs/figures/cm_dt_density_category.png](outputs/figures/cm_dt_density_category.png)
- Models:
  - [outputs/models/knn_map_type.pkl](outputs/models/knn_map_type.pkl)
  - [outputs/models/svm_map_type.pkl](outputs/models/svm_map_type.pkl)
  - [outputs/models/dt_map_type.pkl](outputs/models/dt_map_type.pkl)
  - [outputs/models/bagging_map_type.pkl](outputs/models/bagging_map_type.pkl)
  - [outputs/models/boosting_map_type.pkl](outputs/models/boosting_map_type.pkl)
  - [outputs/models/knn_difficulty.pkl](outputs/models/knn_difficulty.pkl)
  - [outputs/models/svm_difficulty.pkl](outputs/models/svm_difficulty.pkl)
  - [outputs/models/dt_difficulty.pkl](outputs/models/dt_difficulty.pkl)
  - [outputs/models/knn_density_category.pkl](outputs/models/knn_density_category.pkl)
  - [outputs/models/svm_density_category.pkl](outputs/models/svm_density_category.pkl)
  - [outputs/models/dt_density_category.pkl](outputs/models/dt_density_category.pkl)
  - [outputs/models/scaler_phase4.pkl](outputs/models/scaler_phase4.pkl)

## Notes
- Leakage checks passed: no duplicate map entries, no train/test overlap, permutation accuracy ~0.29.
- Top Decision Tree features for map type were symmetry and connectivity metrics (vertical/horizontal symmetry, num_free_components), indicating strong structural cues in the grid-derived features.
- High but not perfect map-type accuracy suggests separability without obvious metadata leakage.
- Near-perfect classification likely reflects deterministic structure in the generators (maze/room/random/street) and highly discriminative structural features (symmetry, connectivity, border patterns). This should be stated explicitly in Phase 6 analysis.
- Clustering vs classification interpretation: KMeans captures coarse navigability groups, while supervised classifiers learn rule-based structure aligned to the labeled map types.
- PCA is used for clustering; downstream inference in Phase 6/7 should consistently apply scaler -> PCA -> KMeans when assigning clusters.
- AdaBoost is not a failed model here; it is slightly less stable under this feature space while still achieving strong accuracy.