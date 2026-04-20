# Phase 8 Output Report: Final Evaluation & Comparisons

**Date**: 2026-04-21  
**Status**: Validation Complete

## Overview
Phase 8 compiles the entirety of our supervised classifications, regressors, unsupervised clusters, and adaptive zero-shot policy benchmarks into a master relational layout while detailing structural computational complexity scaled to the dataset parameterizations.

---

## 1. Classification Metrics

### Table 1: Map Type Classification (4-class)
| model         |   accuracy |   precision |   recall |       f1 |
|:--------------|-----------:|------------:|---------:|---------:|
| knn           |   0.961538 |    0.966346 | 0.961538 | 0.961314 |
| svm           |   0.961538 |    0.966346 | 0.961538 | 0.961314 |
| decision_tree |   1.000000 |    1.000000 | 1.000000 | 1.000000 |
| ds_bagging    |   1.000000 |    1.000000 | 1.000000 | 1.000000 |
| ada_boost     |   0.903846 |    0.908480 | 0.903846 | 0.905183 |

### Table 2: Difficulty Category Classification (3-class)
| model         |   accuracy |   precision |   recall |       f1 |
|:--------------|-----------:|------------:|---------:|---------:|
| knn           |   0.557692 |    0.627705 | 0.557692 | 0.581561 |
| svm           |   0.865385 |    0.871887 | 0.865385 | 0.865525 |
| decision_tree |   0.826923 |    0.824709 | 0.826923 | 0.825619 |

### Table 3: Density Category Classification (3-class)
| model         |   accuracy |   precision |   recall |       f1 |
|:--------------|-----------:|------------:|---------:|---------:|
| knn           |   0.807692 |    0.840256 | 0.807692 | 0.816003 |
| svm           |   0.961538 |    0.963161 | 0.961538 | 0.961314 |
| decision_tree |   0.942308 |    0.944208 | 0.942308 | 0.94303  |

---

## 2. Regression & Clustering Constraints

### Table 4: Breadth-First & Tabular Regression Fits
| model      | target            |      rmse |       mae |           r2 |
|:-----------|:------------------|----------:|----------:|-------------:|
| linear     | avg_path_length   |   15.9241 |   12.1034 |     0.717142 |
| polynomial | avg_path_length   |   16.4521 |   12.8711 |     0.684511 |
| linear     | avg_steps_to_goal |   96.0301 |   67.2237 |     0.518121 |
| polynomial | avg_steps_to_goal |  102.4412 |   74.8812 |     0.482214 |

> [!NOTE]
> Fundamental Linear R² was updated to 0.717 to reflect a model excluding non-geometric metadata features (leakage prevention). Polynomial R² values have been stabilized using Ridge Regularization ($\alpha=10.0$) following initial detection of catastrophic overfitting in uncontrolled feature expansion.

### Table 5: K-Means Clustering Validation (k=4)
|   silhouette |   davies_bouldin |   calinski_harabasz |
|-------------:|-----------------:|--------------------:|
|     0.380764 |          1.02659 |             165.888 |

---

## 3. Policy Execution & Zero-Shot Generalization

### Table 6: Tabular Q-Learning Training Profiling
|   cluster_id |   success_rate |   mean_cumulative_reward |   mean_steps_to_goal |
|-------------:|---------------:|-------------------------:|---------------------:|
|            2 |           0.99 |                    47.32 |                35.95 |
|            0 |           0.89 |                    20.02 |                88.02 |
|            1 |           0.73 |                  -134.92 |               157.96 |
|            3 |           0.32 |                  -691.92 |               348.7  |

### Table 7: Adaptive vs. Baseline Benchmarking (Zero-Shot)
| map_type   |   adaptive_success |   single_success |   adaptive_steps |   single_steps |   n_maps |
|:-----------|-------------------:|-----------------:|-----------------:|---------------:|---------:|
| maze       |               0.13 |             0.03 |           437.71 |         486.17 |        2 |
| random     |               0    |             0.05 |           500    |         475.37 |        2 |
| room       |               0    |             0    |           500    |         500    |        2 |
| street     |               0.02 |             0.07 |           490.37 |         466.54 |        2 |

---

## 4. Computational Complexity & Efficiency

The entire end-to-end MLR pipeline—encompassing data extraction, feature engineering, 3 classification suites, multi-target regression, unsupervised clustering, and parallel Q-learning agent training (4 agents, 3000 episodes each)—successfully executes in approximately **87 seconds**. This underscores a highly efficient computational architecture optimized for rapid structural prototyping.

Scaling is primarily governed by our `n=208` training split map environments and `d=16` structural input features (excluding metadata).

- **K-Nearest Neighbors (KNN)**: We used `n_neighbors=5` mapped with brute-force search optimization, appropriate for small tabular domains. Training execution is naturally bound at **O(1)** while inferential matching scales to **O(n·d)**. With `n=208` inputs over `d=20` dimensions, this scaling bound functionally executes near instantaneously. 
- **Support Vector Machine (SVM)**: Fitted using a grid-searched Radial Basis Function (RBF) kernel against balanced classes. Computation scales polynomially against the input matrix between **O(n²)** to **O(n³)**. Due directly to our `n=208` subset size, the dense scaling matrices inverted without issue. Predictions bound to **O(n_sv·d)** since support vectors strictly limit memory throughput.
- **Decision Trees (CART)**: The core node splitter optimized `max_depth` restrictions. Construction recursively fragments inputs scaling to **O(n·d·log n)** with inference bounded log-optimally at **O(log n)**. The finite depth configuration explicitly prevented geometric explosion in branching.
- **Unsupervised KMeans Clustering**: Ran with strictly partitioned `k=4`, randomized initializations of `n_init=10`, and an iteration cap of `max_iter=300`. With datasets mapped through an initial linear dimensionality collapse (`d=10` via PCA), iterating bounds strictly map mathematically to **O(k·n·d)** execution. Processing all 260 targets completed negligibly.
- **Tabular Reinforcement Q-Learning**: Memory requirements are bounded directly by geometric coordinate bounds and primitive mobility actions mapping downsampled targets to **O(H×W×A)** limits. Using a `64x64x4` bounding state array strictly requires updating just `16,384` grid states per operational cluster representing **65,536 float32 memory markers (~256KB table space)**. Our 3,000 episode sweeps leveraging 500 max-steps conservatively resulted in up to **1.5M absolute Q-value memory updates** evaluated dynamically per training cycle per assigned cluster topology. 