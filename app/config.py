from pathlib import Path

# ==========================
# Project Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "trained_models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Dataset Paths
# ==========================

QUESTIONNAIRE_DATA_PATH = DATA_DIR / "questionnaire_normal.csv"
GAME_DATA_PATH = DATA_DIR / "game_normal.csv"

# ==========================
# Model Paths
# ==========================

QUESTIONNAIRE_MODEL_PATH = MODELS_DIR / "questionnaire_iforest.pkl"
QUESTIONNAIRE_SCALER_PATH = MODELS_DIR / "questionnaire_scaler.pkl"

GAME_MODEL_PATH = MODELS_DIR / "game_iforest.pkl"
GAME_SCALER_PATH = MODELS_DIR / "game_scaler.pkl"

# ==========================
# Isolation Forest Settings
# ==========================

QUESTIONNAIRE_N_ESTIMATORS = 300
GAME_N_ESTIMATORS = 500

QUESTIONNAIRE_CONTAMINATION = 0.08
GAME_CONTAMINATION = 0.10

RANDOM_STATE = 42

# ==========================
# Risk Thresholds
# ==========================

LOW_RISK_THRESHOLD = 0.30
MEDIUM_RISK_THRESHOLD = 0.60

# ==========================
# API Metadata
# ==========================

API_TITLE = "NeuroVice AI Engine"
API_VERSION = "1.0.0"

API_DESCRIPTION = """
Neurodevelopmental behavioral anomaly detection engine.
Currently optimized for ADHD behavioral risk analysis.
"""

# ==========================
# Supported Conditions
# ==========================

SUPPORTED_CONDITIONS = [
    "ADHD"
]