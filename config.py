from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_STATE = 42  # Global seed for all stochastic operations.
np.random.seed(RANDOM_STATE)

# -----------------------------------------------------------------------------
# Paths and directory layout
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # Project root.
DATA_DIR = BASE_DIR / "data"  # Root data directory.
RAW_DATA_DIR = DATA_DIR / "raw"  # Extracted .map files.
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Feature/label outputs.
OUTPUTS_DIR = BASE_DIR / "outputs"  # Root for all generated artifacts.
FIGURES_DIR = OUTPUTS_DIR / "figures"  # Plots and visualizations.
MODELS_DIR = OUTPUTS_DIR / "models"  # Serialized models and arrays.
REPORTS_DIR = OUTPUTS_DIR / "reports"  # CSV/MD summaries.
LOGS_DIR = BASE_DIR / "logs"  # Runtime logs.
DATASET_DIR = BASE_DIR / "dataset"  # Optional raw archives.

# Ensure expected directories exist to avoid path errors at runtime.
for _d in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Global experiment settings
# -----------------------------------------------------------------------------
TEST_SIZE = 0.2  # Train/test split proportion.
PCA_COMPONENTS = 10  # Number of PCA components for clustering.
K_NEIGHBORS = 5  # KNN classifier neighbors.

# -----------------------------------------------------------------------------
# Clustering configuration
# -----------------------------------------------------------------------------
N_CLUSTERS = 4  # Expected number of clusters for interpretability.
BEST_K = 2  # Silhouette-optimal k from k selection analysis.
DOWNSTREAM_CLUSTER_K = 4  # k used for RL and policy switching.

# -----------------------------------------------------------------------------
# Labeling thresholds
# -----------------------------------------------------------------------------
DIFFICULTY_LOW_PERCENTILE = 0.33  # Lower percentile for difficulty binning.
DIFFICULTY_HIGH_PERCENTILE = 0.67  # Upper percentile for difficulty binning.
DENSITY_LOW_THRESHOLD = 0.20  # Obstacle density threshold for "low".
DENSITY_HIGH_THRESHOLD = 0.35  # Obstacle density threshold for "high".

# -----------------------------------------------------------------------------
# BFS regression targets
# -----------------------------------------------------------------------------
BFS_SAMPLES = 50  # Random start-goal pairs per map.

# -----------------------------------------------------------------------------
# Tabular RL training and evaluation
# -----------------------------------------------------------------------------
Q_LEARNING_EPISODES = 3000  # Episodes per cluster.
LEARNING_RATE = 0.1  # Q-learning step size.
DISCOUNT_FACTOR = 0.95  # Gamma for future reward discounting.
EPSILON = 1.0  # Initial exploration probability.
EPSILON_DECAY = 0.9985  # Multiplicative decay per episode.
EPSILON_MIN = 0.01  # Minimum exploration floor.
RL_LOG_EVERY = 100  # Episode interval for progress logs.
RL_STEP_LOG_INTERVAL = 100000  # Step interval for detailed logging.
RL_EARLY_STOP_WINDOW = 100  # Rolling window for early-stop check.
RL_EARLY_STOP_PATIENCE = 500  # Episodes without improvement before stopping.
RL_EARLY_STOP_MIN_DELTA = 1.0  # Minimum reward improvement to reset patience.
RL_MAX_STEPS = 500  # Per-episode step cap.
RL_MAX_TRAIN_MAP_DIM = 512  # Maximum original grid size for training.

# Adaptive downsampling targets (see downsample_grid_adaptive).
RL_GRID_DIM_512 = 64  # 512x512 maps -> 64x64 (stride 8).
RL_GRID_DIM_1024 = 64  # 1024x1024 maps -> 64x64 (stride 16).
RL_GRID_DIM_256 = 256  # 256x256 maps -> no downsample.
RL_GRID_DIM_FINE = 128  # Narrow maps -> 128x128 (stride 4 on 512).
RL_MIN_FREE_CELLS = 20  # Skip map if too few free cells after downsample.
RL_MAX_GRID_DIM = 64  # Legacy fallback grid size for RL.

# Curriculum and start-state configuration.
RL_START_NEAR_GOAL_PROB = 0.0  # Probability of near-goal starts.
RL_NEAR_GOAL_RADIUS = 5  # Radius for near-goal sampling.
RL_CURRICULUM_P1 = 0.0  # Phase 1 schedule fraction.
RL_CURRICULUM_P2 = 0.0  # Phase 2 schedule fraction.
RL_CURRICULUM_PROB_EARLY = 0.0  # Near-goal probability in early training.
RL_CURRICULUM_PROB_MID = 0.0  # Near-goal probability in mid training.
RL_CURRICULUM_PROB_LATE = 0.0  # Near-goal probability in late training.

# Reward shaping.
RL_REWARD_GOAL = 100.0  # Reward for reaching the goal.
RL_REWARD_STEP = -1.0  # Per-step penalty to encourage shorter paths.
RL_REWARD_INVALID = -10.0  # Penalty for invalid moves.
RL_REWARD_TIMEOUT = -100.0  # Penalty for episode timeout.
RL_REWARD_DISTANCE_SCALE = 1.0  # Scale for distance-based shaping term.

# -----------------------------------------------------------------------------
# Dataset metadata
# -----------------------------------------------------------------------------
MAP_TYPE_CLASSES = ["maze", "room", "random", "street"]  # 4-class label set.
STREET_CITIES = [
    "Berlin",
    "Boston",
    "Denver",
    "London",
    "Milan",
    "Moscow",
    "NewYork",
    "Paris",
    "Shanghai",
    "Sydney",
]  # City names encoded in street map filenames.
