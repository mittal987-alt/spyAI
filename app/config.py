
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

# -------------------------------------------------------------------
# Security & Environment Configuration
# -------------------------------------------------------------------
MODELAUTOPSY_ENV = os.getenv("MODELAUTOPSY_ENV", "development").strip().lower()
IS_PRODUCTION = MODELAUTOPSY_ENV == "production"

# Optional API key for sensitive operations
MODELAUTOPSY_API_KEY = os.getenv("MODELAUTOPSY_API_KEY", "").strip() or None

# CORS allowed origins (comma-separated in env)
_DEFAULT_CORS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", _DEFAULT_CORS).split(",")
    if origin.strip()
]

# Allowed hostnames for TrustedHostMiddleware
_DEFAULT_HOSTS = "localhost,127.0.0.1,testserver"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("ALLOWED_HOSTS", _DEFAULT_HOSTS).split(",")
    if host.strip()
]

# Allow local webhook destinations (loopback/private IP).
# Defaults to False for SSRF protection; can be enabled in isolated dev/tests.
WEBHOOK_ALLOW_LOCAL = os.getenv(
    "MODELAUTOPSY_WEBHOOK_ALLOW_LOCAL", "false"
).strip().lower() in ("true", "1", "yes")

# Max file upload constraints
MAX_UPLOAD_SIZE_BYTES = int(
    os.getenv("MODELAUTOPSY_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))  # 10 MB
)
MAX_DATASET_ROWS = int(os.getenv("MODELAUTOPSY_MAX_ROWS", "50000"))
MAX_DATASET_COLUMNS = int(os.getenv("MODELAUTOPSY_MAX_COLUMNS", "200"))

# -------------------------------------------------------------------
# Rate Limiting
# -------------------------------------------------------------------
RATE_LIMIT_REQUESTS = int(os.getenv("MODELAUTOPSY_RATE_LIMIT", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("MODELAUTOPSY_RATE_WINDOW", "60"))

# -------------------------------------------------------------------
# Webhook Settings
# -------------------------------------------------------------------
WEBHOOK_TIMEOUT_SECONDS = int(os.getenv("MODELAUTOPSY_WEBHOOK_TIMEOUT", "10"))
WEBHOOK_MAX_RETRIES = int(os.getenv("MODELAUTOPSY_WEBHOOK_MAX_RETRIES", "3"))
WEBHOOK_MAX_PAYLOAD_BYTES = int(
    os.getenv("MODELAUTOPSY_WEBHOOK_MAX_PAYLOAD", str(1 * 1024 * 1024))  # 1 MB
)

# -------------------------------------------------------------------
# Deep Learning / Training
# -------------------------------------------------------------------
BATCH_SIZE = int(os.getenv("MODELAUTOPSY_BATCH_SIZE", "32"))
MAX_EPOCHS = int(os.getenv("MODELAUTOPSY_MAX_EPOCHS", "50"))
LEARNING_RATE = float(os.getenv("MODELAUTOPSY_LEARNING_RATE", "0.001"))

# -------------------------------------------------------------------
# RAG & LLM Settings
# -------------------------------------------------------------------
RAG_DB_PATH = PROJECT_ROOT / "rag_history.db"
RAG_MAX_HISTORY = int(os.getenv("MODELAUTOPSY_RAG_MAX_HISTORY", "100"))
LLM_MODEL_NAME = os.getenv("MODELAUTOPSY_LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
LOG_LEVEL = os.getenv("MODELAUTOPSY_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("MODELAUTOPSY_LOG_FORMAT", "text")  # "json" or "text"

# -------------------------------------------------------------------
# Reports & Exports
# -------------------------------------------------------------------
EXPORT_FORMATS = ["csv", "json", "pdf"]
MAX_REPORT_ROWS = int(os.getenv("MODELAUTOPSY_MAX_REPORT_ROWS", "10000"))

# -------------------------------------------------------------------
# Alert Retention & Pagination
# -------------------------------------------------------------------
ALERTS_MAX_RETENTION_DAYS = int(os.getenv("MODELAUTOPSY_ALERTS_RETENTION_DAYS", "30"))
ALERTS_MAX_PER_PAGE = int(os.getenv("MODELAUTOPSY_ALERTS_PER_PAGE", "50"))

# -------------------------------------------------------------------
# Smart Alert Rules Engine
# -------------------------------------------------------------------
# SQLite path for persisted alert rules
ALERT_RULES_DB_PATH = PROJECT_ROOT / "alert_rules.db"
# Maximum conditions per rule
ALERT_RULES_MAX_CONDITIONS = int(os.getenv("MODELAUTOPSY_ALERT_RULES_MAX_CONDITIONS", "10"))
# Maximum stored rules
ALERT_RULES_MAX_RULES = int(os.getenv("MODELAUTOPSY_ALERT_RULES_MAX_RULES", "100"))


def validate_production_config() -> list[str]:
    """Validate critical security settings in production mode."""
    warnings: list[str] = []

    if IS_PRODUCTION:
        if "*" in CORS_ALLOWED_ORIGINS:
            raise ValueError(
                "Wildcard '*' origin is prohibited in production when credentials are enabled."
            )
        if "*" in ALLOWED_HOSTS:
            warnings.append(
                "ALLOWED_HOSTS contains wildcard '*'. Restrict to explicit hostnames in production."
            )
        if WEBHOOK_ALLOW_LOCAL:
            warnings.append(
                "WEBHOOK_ALLOW_LOCAL is enabled in production. This poses an SSRF risk."
            )
        if not MODELAUTOPSY_API_KEY:
            warnings.append(
                "MODELAUTOPSY_API_KEY is not set. Sensitive endpoints remain unauthenticated."
            )

    return warnings
