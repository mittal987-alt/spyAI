
import os
from pathlib import Path

# Project root: C:\projects\spyai
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
ARTIFACTS_DIR = Path(
    os.getenv(
        "MODELAUTOPSY_ARTIFACTS_DIR",
        str(PROJECT_ROOT / "artifacts"),
    )
).resolve()

REPORTS_DIR = Path(
    os.getenv(
        "MODELAUTOPSY_REPORTS_DIR",
        str(PROJECT_ROOT / "reports"),
    )
).resolve()

# Reproducibility
RANDOM_STATE = 42
N_SAMPLES = 6000
N_FEATURES = 8
TEST_SIZE = 0.25

# Drift detection
DRIFT_ALPHA = 0.05
DRIFT_EFFECT_THRESHOLD = 0.10
DRIFT_MEDIUM_THRESHOLD = 0.10
DRIFT_HIGH_THRESHOLD = 0.30

# MLflow experiment tracking
MLFLOW_EXPERIMENT_NAME = "ModelAutopsy-Drift-Experiments"

# SQLite-backed tracking database
MLFLOW_TRACKING_URI = (
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)

# Persistent drift alerts database
ALERTS_DB_PATH = PROJECT_ROOT / "alerts.db"
EARLY_STOPPING_PATIENCE = 5
CHECKPOINT_DIR = ARTIFACTS_DIR / "deep_learning"
