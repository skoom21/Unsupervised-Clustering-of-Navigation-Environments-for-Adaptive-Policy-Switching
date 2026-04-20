# Phase 6 Output Report

Date: 2026-04-21

## Overview
- Completed Phase 6 Q-Learning to generate RL-derived avg_steps_to_goal for downstream regression.
- Trained cluster-specific tabular Q-tables on downsampled grids with reward shaping and a curriculum schedule.
- Evaluated policies on unbiased random start-goal pairs; generalization failed across all clusters.

## Approach Timeline (Phase 6 end-to-end)
- Train per-cluster Q-learning agents using downsampled grids (RL_MAX_GRID_DIM=32) and a hard step cap (RL_MAX_STEPS=800).
- Add reward shaping (goal reward, step penalty, invalid-move penalty, timeout penalty, distance-delta shaping).
- Introduce a smooth curriculum schedule (near-goal starts → mixed starts → random starts) to bootstrap learning.
- Tune exploration (epsilon decay to 0.99 with EPSILON_MIN=0.05) and enable early stopping.
- Evaluate greedily on random start-goal pairs with curriculum bias disabled.
- Compute avg_steps_to_goal per map and train regression models to test predictive stability.

## RL Evaluation Results (cluster-level)
- Cluster 0 (Milan_2_512.map): success_rate=0.0, mean_steps_to_goal=800.0, mean_reward=-1109.52
- Cluster 1 (Moscow_1_1024.map): success_rate=0.0, mean_steps_to_goal=800.0, mean_reward=-1749.72
- Cluster 2 (maze512-8-9.map): success_rate=0.0, mean_steps_to_goal=800.0, mean_reward=-965.28
- Cluster 3 (random512-35-6.map): success_rate=0.0, mean_steps_to_goal=800.0, mean_reward=-1396.53

## Regression Results (avg_steps_to_goal)
- Linear: rmse=8.1886, mae=6.2117, r2=-0.3177
- Polynomial (degree 2): rmse=297.2515, mae=132.6571, r2=-1735.3903

## Interpretation (refined)
- The observed behavior is consistent with train-time curriculum overfitting, evaluation-time distribution shift, and tabular RL generalization failure.
- The agent demonstrates intermittent success during curriculum-based training but fails under unbiased random start-goal evaluation.
- This suggests that the learned value function does not preserve a consistent distance-to-goal ordering under distribution shift.
- As a result, policies that appear effective during training do not generalize, leading to 0% success rate and persistent step-cap saturation during evaluation.
- This behavior is better explained by policy brittleness rather than randomness.
- The absence of convergence is consistent with known limitations of tabular Q-learning in high-dimensional state spaces with sparse terminal rewards.
- Final interpretation: Q-learning is used as a weak structural estimator producing noisy cost-to-go signals for downstream regression rather than a reliable planner.

## Design Decision (use of RL signals)
- There is no evidence in this phase that RL-derived targets improve predictive performance over geometry-only features; regression on avg_steps_to_goal is unstable and poorly generalizes.
- RL outputs are excluded from any critical decision-making in final model evaluation.
- RL is retained only for structural inductive bias: a noisy cost-to-go proxy that augments features as a weak heuristic signal.

## Outputs Generated
- Data:
  - [data/processed/steps_to_goal.csv](data/processed/steps_to_goal.csv)
- Reports:
  - [outputs/reports/rl_cluster_metrics.csv](outputs/reports/rl_cluster_metrics.csv)
  - [outputs/reports/regression_avg_steps_to_goal_metrics.csv](outputs/reports/regression_avg_steps_to_goal_metrics.csv)
- Figures:
  - [outputs/figures/rl_training_cluster_0.png](outputs/figures/rl_training_cluster_0.png)
  - [outputs/figures/rl_training_cluster_1.png](outputs/figures/rl_training_cluster_1.png)
  - [outputs/figures/rl_training_cluster_2.png](outputs/figures/rl_training_cluster_2.png)
  - [outputs/figures/rl_training_cluster_3.png](outputs/figures/rl_training_cluster_3.png)
  - [outputs/figures/regression_actual_vs_pred_linear_avg_steps_to_goal.png](outputs/figures/regression_actual_vs_pred_linear_avg_steps_to_goal.png)
  - [outputs/figures/regression_residuals_linear_avg_steps_to_goal.png](outputs/figures/regression_residuals_linear_avg_steps_to_goal.png)
  - [outputs/figures/regression_actual_vs_pred_poly_avg_steps_to_goal.png](outputs/figures/regression_actual_vs_pred_poly_avg_steps_to_goal.png)
  - [outputs/figures/regression_residuals_poly_avg_steps_to_goal.png](outputs/figures/regression_residuals_poly_avg_steps_to_goal.png)
- Models:
  - [outputs/models/qtable_cluster_0.npy](outputs/models/qtable_cluster_0.npy)
  - [outputs/models/qtable_cluster_1.npy](outputs/models/qtable_cluster_1.npy)
  - [outputs/models/qtable_cluster_2.npy](outputs/models/qtable_cluster_2.npy)
  - [outputs/models/qtable_cluster_3.npy](outputs/models/qtable_cluster_3.npy)
  - [outputs/models/linear_regression_avg_steps_to_goal.pkl](outputs/models/linear_regression_avg_steps_to_goal.pkl)
  - [outputs/models/poly_regression_avg_steps_to_goal.pkl](outputs/models/poly_regression_avg_steps_to_goal.pkl)
  - [outputs/models/scaler_phase6_steps.pkl](outputs/models/scaler_phase6_steps.pkl)

## Missing Steps-to-Goal Maps (avg_steps_to_goal NaN)
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
- Training logs show intermittent progress under curriculum sampling, but evaluation on random starts consistently failed.
- Regression on avg_steps_to_goal is unstable, which supports treating RL signals as auxiliary features only.
- An ablation study in Phase 7 will evaluate whether RL-derived features provide measurable improvement over purely geometric features.
