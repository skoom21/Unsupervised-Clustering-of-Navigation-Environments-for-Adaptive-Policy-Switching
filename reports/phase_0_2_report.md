# Phase 0-2 Output Report

Date: 2026-04-20

## Overview
- Completed Phases 0, 1A, 1B, and 2 per project prompt.
- Dataset extraction verified 260 maps across 4 types.
- Feature extraction, labeling, validation, and preprocessing outputs generated.

## Phase 0: Setup and Extraction
- Archive extraction status: OK
- Map counts verified:
  - maze: 60
  - room: 40
  - random: 70
  - street: 90
  - total: 260

## Phase 1A: Feature Extraction
- Loaded maps: 260
- Feature matrix: 260 rows x 17 columns
- Null counts: all zero
- Sample map visualization created.

## Phase 1B: Label Generation
- Map type distribution: street=90, random=70, maze=60, room=40
- Difficulty distribution: medium=89, easy=86, hard=85
- Density category distribution: medium=150, low=80, high=30
- Random map obstacle correlation (known vs computed): r=0.928

Validation checks:
- Random maps with known density <= 0.15 classified low: PASS (100%)
- Random maps with known density >= 0.35 classified high: PASS (100%)
- Maze corridor width 1 classified hard: PASS (100%)
- Maze corridor width 32 classified easy: PASS (100%)
- Random density correlation r=0.928: PASS

## Phase 2: Preprocessing
- Numeric feature matrix: 260 rows x 20 columns
- Missing values: none
- StandardScaler fitted and saved
- PCA fitted and saved (10 components)
- Train/test splits created for:
  - label_map_type
  - label_difficulty
  - label_density_category

Explained variance (cumulative):
- PC1: 0.3867
- PC2: 0.6561
- PC3: 0.7638
- PC4: 0.8361
- PC5: 0.8949
- PC6: 0.9290
- PC7: 0.9547
- PC8: 0.9713
- PC9: 0.9859
- PC10: 0.9914

## Outputs Generated
- Features: [data/processed/raw_features.csv](data/processed/raw_features.csv)
- Labeled dataset: [data/processed/labeled_dataset.csv](data/processed/labeled_dataset.csv)
- Sample maps: [outputs/figures/sample_maps_by_type.png](outputs/figures/sample_maps_by_type.png)
- Label validation: [outputs/reports/label_validation.txt](outputs/reports/label_validation.txt)
- Scaler: [outputs/models/scaler.pkl](outputs/models/scaler.pkl)
- PCA model: [outputs/models/pca_reducer.pkl](outputs/models/pca_reducer.pkl)
- PCA variance plot: [outputs/figures/pca_variance.png](outputs/figures/pca_variance.png)

## Notes
- Density-category validation now passes after aligning thresholds with dataset max density (0.40).
- Threshold update applied in code: `DENSITY_HIGH_THRESHOLD = 0.35`.
- Fix relevance: it corrects mislabeling of 0.35–0.40 density maps as medium, improving label fidelity for downstream clustering and classification.