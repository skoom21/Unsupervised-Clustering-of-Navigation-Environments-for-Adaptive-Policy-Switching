# Phase 5 Output Report

Date: 2026-04-20

## Overview
- Completed Phase 5 regression using BFS-estimated average path lengths as target.
- BFS ran on 4x downsampled grids with 50 sampled start-goal pairs per map.
- Regression models trained and evaluated; plots and metrics saved.
- BFS-derived columns were removed from inputs to prevent leakage.

## BFS Path Lengths
- Total maps processed: 260
- Missing avg_path_length values: 41
- Valid BFS coverage: 219 / 260 (~84.2%)
- Likely causes: fewer than 10 free cells after downsampling or no reachable pairs sampled.

## Regression Results (avg_path_length)
- Linear: rmse=14.0663, mae=9.4923, r2=0.9225
- Polynomial (degree 2): rmse=956.6529, mae=462.4750, r2=-357.5548

## Outputs Generated
- Data:
  - [data/processed/path_lengths.csv](data/processed/path_lengths.csv)
- Reports:
  - [outputs/reports/regression_avg_path_length_metrics.csv](outputs/reports/regression_avg_path_length_metrics.csv)
- Figures:
  - [outputs/figures/regression_actual_vs_pred_linear_avg_path_length.png](outputs/figures/regression_actual_vs_pred_linear_avg_path_length.png)
  - [outputs/figures/regression_residuals_linear_avg_path_length.png](outputs/figures/regression_residuals_linear_avg_path_length.png)
  - [outputs/figures/regression_actual_vs_pred_poly_avg_path_length.png](outputs/figures/regression_actual_vs_pred_poly_avg_path_length.png)
  - [outputs/figures/regression_residuals_poly_avg_path_length.png](outputs/figures/regression_residuals_poly_avg_path_length.png)
- Models:
  - [outputs/models/linear_regression_avg_path_length.pkl](outputs/models/linear_regression_avg_path_length.pkl)
  - [outputs/models/poly_regression_avg_path_length.pkl](outputs/models/poly_regression_avg_path_length.pkl)
  - [outputs/models/scaler_phase5.pkl](outputs/models/scaler_phase5.pkl)

## Missing Path Length Maps (avg_path_length NaN)
- [data/raw/maze/maze512-1-0.map](data/raw/maze/maze512-1-0.map)
- [data/raw/maze/maze512-1-1.map](data/raw/maze/maze512-1-1.map)
- [data/raw/maze/maze512-1-2.map](data/raw/maze/maze512-1-2.map)
- [data/raw/maze/maze512-1-3.map](data/raw/maze/maze512-1-3.map)
- [data/raw/maze/maze512-1-4.map](data/raw/maze/maze512-1-4.map)
- [data/raw/maze/maze512-1-5.map](data/raw/maze/maze512-1-5.map)
- [data/raw/maze/maze512-1-6.map](data/raw/maze/maze512-1-6.map)
- [data/raw/maze/maze512-1-7.map](data/raw/maze/maze512-1-7.map)
- [data/raw/maze/maze512-1-8.map](data/raw/maze/maze512-1-8.map)
- [data/raw/maze/maze512-1-9.map](data/raw/maze/maze512-1-9.map)
- [data/raw/random/random512-40-0.map](data/raw/random/random512-40-0.map)
- [data/raw/random/random512-40-1.map](data/raw/random/random512-40-1.map)
- [data/raw/random/random512-40-6.map](data/raw/random/random512-40-6.map)
- [data/raw/random/random512-40-7.map](data/raw/random/random512-40-7.map)
- [data/raw/room/16room_000.map](data/raw/room/16room_000.map)
- [data/raw/room/16room_001.map](data/raw/room/16room_001.map)
- [data/raw/room/16room_002.map](data/raw/room/16room_002.map)
- [data/raw/room/16room_003.map](data/raw/room/16room_003.map)
- [data/raw/room/16room_004.map](data/raw/room/16room_004.map)
- [data/raw/room/16room_005.map](data/raw/room/16room_005.map)
- [data/raw/room/16room_006.map](data/raw/room/16room_006.map)
- [data/raw/room/16room_007.map](data/raw/room/16room_007.map)
- [data/raw/room/16room_008.map](data/raw/room/16room_008.map)
- [data/raw/room/16room_009.map](data/raw/room/16room_009.map)
- [data/raw/room/32room_000.map](data/raw/room/32room_000.map)
- [data/raw/room/32room_002.map](data/raw/room/32room_002.map)
- [data/raw/room/32room_003.map](data/raw/room/32room_003.map)
- [data/raw/room/32room_004.map](data/raw/room/32room_004.map)
- [data/raw/room/32room_009.map](data/raw/room/32room_009.map)
- [data/raw/room/64room_000.map](data/raw/room/64room_000.map)
- [data/raw/room/64room_009.map](data/raw/room/64room_009.map)
- [data/raw/room/8room_000.map](data/raw/room/8room_000.map)
- [data/raw/room/8room_001.map](data/raw/room/8room_001.map)
- [data/raw/room/8room_002.map](data/raw/room/8room_002.map)
- [data/raw/room/8room_003.map](data/raw/room/8room_003.map)
- [data/raw/room/8room_004.map](data/raw/room/8room_004.map)
- [data/raw/room/8room_005.map](data/raw/room/8room_005.map)
- [data/raw/room/8room_006.map](data/raw/room/8room_006.map)
- [data/raw/room/8room_007.map](data/raw/room/8room_007.map)
- [data/raw/room/8room_008.map](data/raw/room/8room_008.map)
- [data/raw/room/8room_009.map](data/raw/room/8room_009.map)

## Notes
- Missing avg_path_length rows were excluded from regression training by filtering to non-null targets.
- Regression was trained on ~84% of maps with valid connectivity under downsampled BFS.
- BFS-derived columns (avg_path_length, max/min path length, reachable-pair count) were removed from the feature set before training.
- Linear regression is the trustworthy baseline here (good fit with noisy BFS targets).
- Polynomial regression is unstable on this feature set (overfitting/scale sensitivity), so its metrics should be discarded.
- Missing values cluster around maze512-1-*, random512-40-*, and room* maps, suggesting downsampling or sparse connectivity effects.
- Mitigation ideas: adaptive BFS (retry at higher resolution), multi-resolution grids, or a fallback heuristic for sparse maps.
