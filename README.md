# Unsupervised Clustering of Navigation Environments for Adaptive Policy Switching

## Project Complete Delivery

This repository contains an end-to-end Machine Learning pipeline developed to extract, standardize, classify, cluster, and ultimately reinforcement-learn zero-shot optimal policies applied across geographically generated octile grid maps (mazes, randomly generated fields, constructed interior rooms, and macro-streets environments).

### Pipeline Execution Requirements
**Language**: Python 3.x
**Libraries**: pandas, numpy, scikit-learn, matplotlib, scipy, tabulate

It is strictly recommended you execute within an isolated virtual environment (`.venv`).

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tabulate
```

### Full Pipeline Automation
The entire extraction, encoding, training, benchmarking, and formatting process is dynamically mapped from `Phase 0` directly to `Phase 8` using the main runner.

```bash
python main.py
```
*Note*: Full Q-Learning tabular propagation against the high-dimensional grid matrices may require significant computational wall-time. Due to strict random seeding integrated natively across algorithms, all executed modeling paths are completely reproducible.

### Deliverables Output Architecture
Executing `main.py` structures explicit geometric and tabular analytics directly to disk:
*   `data/` raw archives mapping into standard binary octile matrices directly parsed, evaluated via parallel components extraction into `processed/raw_features.csv`.
*   `outputs/models/` retains all classification nodes, regression curves, clustered KMeans groupings, PCA dimensions, normalized bounds matrices, and `.npy` state-space Q-tables for all environments.
*   `outputs/figures/` exports explicitly bounded plotting graphics confirming analytical constraints.
*   `outputs/reports/` details all markdown documentation generated structurally during each milestone.

**See `outputs/reports/phase_8_report.md` for explicit performance aggregation across Map Type Clustering, Difficulty Scaling, R2 Regressions mapping Structural Complexity, tabulate RL limits, and Zero-Shot Inferential Scaling computations.**
