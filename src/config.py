"""
FabSight Configuration
Configuration constants and environment variables
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── API Configuration ──────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_MAIN = "gpt-4o-mini"
OPENAI_MODEL_REPORT = "gpt-4o"
OPENAI_MAX_TOKENS = 800
OPENAI_TEMPERATURE = 0.3

# ── API Retry Configuration ────────────────────────
OPENAI_RETRY_ATTEMPTS = 3
OPENAI_RETRY_DELAY = 1
OPENAI_RETRY_MAX_DELAY = 10
OPENAI_TIMEOUT = 30

# ── Data Configuration ─────────────────────────────
DATA_DIR = "data/raw"
DATA_X_PATH = os.path.join(DATA_DIR, "X_processed.csv")
DATA_Y_PATH = os.path.join(DATA_DIR, "y.csv")
TOP5_SENSORS_PATH = os.path.join(DATA_DIR, "top5_sensors.csv")
OPERATION_LOG_PATH = os.path.join(DATA_DIR, "operation_log.csv")
REPORT_PATH = os.path.join(DATA_DIR, "report.txt")
AGENT_REPORT_PATH = os.path.join(DATA_DIR, "agent_report.txt")

# ── Model Configuration ────────────────────────────
ISOLATION_FOREST_N_ESTIMATORS = 200
ISOLATION_FOREST_CONTAMINATION = 0.07
ISOLATION_FOREST_N_JOBS = -1
ISOLATION_FOREST_RANDOM_STATE = 42

GRADIENT_BOOSTING_N_ESTIMATORS = 300
GRADIENT_BOOSTING_MAX_DEPTH = 3
GRADIENT_BOOSTING_LEARNING_RATE = 0.05
GRADIENT_BOOSTING_SUBSAMPLE = 0.8
GRADIENT_BOOSTING_MIN_SAMPLES_LEAF = 5
GRADIENT_BOOSTING_RANDOM_STATE = 42

PCA_N_COMPONENTS = 50
PCA_RANDOM_STATE = 42

# ── Streamlit Configuration ────────────────────────
STREAMLIT_MAX_API_CALLS = 10
STREAMLIT_PAGE_TITLE = "FabSight - Smart Fab AI Platform"
STREAMLIT_PAGE_ICON = "🔬"

# ── Anomaly Detection Thresholds ───────────────────
ANOMALY_HIGH_RISK_THRESHOLD = 0.7
ANOMALY_MEDIUM_RISK_THRESHOLD = 0.4

# ── Process Configuration ──────────────────────────
PROCESS_THRESHOLDS = {
    "CVD": {"warning": 0.6, "critical": 0.8},
    "ETCH": {"warning": 0.6, "critical": 0.8},
    "CMP": {"warning": 0.5, "critical": 0.75},
    "LITHO": {"warning": 0.55, "critical": 0.78},
}

PROCESS_ORDER = ["LITHO", "CVD", "ETCH", "CMP"]

# ── UI Configuration ──────────────────────────────
PROCESS_COLORS = {
    "CVD": "#ff4b4b",
    "ETCH": "#ffa500",
    "CMP": "#4a90d9",
    "LITHO": "#4ad94a",
}

ALERT_COLOR_CRITICAL = "#ff4b4b"
ALERT_COLOR_WARNING = "#ffa500"
ALERT_COLOR_NORMAL = "#00cc44"

# ── Logging Configuration ──────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = "logs"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
