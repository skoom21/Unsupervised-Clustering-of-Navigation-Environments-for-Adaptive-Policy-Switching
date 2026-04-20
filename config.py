from pathlib import Path
import numpy as np

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
DATASET_DIR = BASE_DIR / "dataset"

# Global initialization of output boundaries avoiding all path errors
for _d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 4
BEST_K = 2
DOWNSTREAM_CLUSTER_K = 4
RANDOM_STATE = 42
TEST_SIZE = 0.2
PCA_COMPONENTS = 10
K_NEIGHBORS = 5
Q_LEARNING_EPISODES = 3000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.9985
EPSILON_MIN = 0.01
RL_LOG_EVERY = 100
RL_STEP_LOG_INTERVAL = 100000
RL_EARLY_STOP_WINDOW = 100
RL_EARLY_STOP_PATIENCE = 500
RL_EARLY_STOP_MIN_DELTA = 1.0
RL_MAX_STEPS = 500
RL_MAX_TRAIN_MAP_DIM = 512
# Adaptive downsampling targets (see downsample_grid_adaptive)
RL_GRID_DIM_512 = 64         # 512x512 maps  -> 64x64  (stride 8)
RL_GRID_DIM_1024 = 64        # 1024x1024 maps -> 64x64  (stride 16)
RL_GRID_DIM_256 = 256        # 256x256 maps  -> no downsample
RL_GRID_DIM_FINE = 128       # narrow-corridor / few-room maps -> 128x128 (stride 4 on 512)
RL_MIN_FREE_CELLS = 20       # skip map in eval if fewer free cells after downsample
RL_MAX_GRID_DIM = 64         # legacy fallback
RL_START_NEAR_GOAL_PROB = 0.0
RL_NEAR_GOAL_RADIUS = 5
RL_CURRICULUM_P1 = 0.0
RL_CURRICULUM_P2 = 0.0
RL_CURRICULUM_PROB_EARLY = 0.0
RL_CURRICULUM_PROB_MID = 0.0
RL_CURRICULUM_PROB_LATE = 0.0
RL_REWARD_GOAL = 100.0
RL_REWARD_STEP = -1.0
RL_REWARD_INVALID = -10.0
RL_REWARD_TIMEOUT = -100.0
RL_REWARD_DISTANCE_SCALE = 1.0
BFS_SAMPLES = 50
DIFFICULTY_LOW_PERCENTILE = 0.33
DIFFICULTY_HIGH_PERCENTILE = 0.67
DENSITY_LOW_THRESHOLD = 0.20
DENSITY_HIGH_THRESHOLD = 0.35

MAP_TYPE_CLASSES = ["maze", "room", "random", "street"]
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
]
