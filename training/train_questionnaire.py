import sys
from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# Enable app imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.dataset_loader import load_questionnaire_dataset
from app.config import (
    QUESTIONNAIRE_MODEL_PATH,
    QUESTIONNAIRE_SCALER_PATH,
    QUESTIONNAIRE_N_ESTIMATORS,
    QUESTIONNAIRE_CONTAMINATION,
    RANDOM_STATE,
)


def train_questionnaire_model():
    """
    Train questionnaire anomaly detection model.
    """

    print("Loading questionnaire dataset...")

    df = load_questionnaire_dataset()

    print(f"Loaded {len(df)} questionnaire samples")

    X = df.values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=QUESTIONNAIRE_N_ESTIMATORS,
        contamination=QUESTIONNAIRE_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training questionnaire Isolation Forest...")

    model.fit(X_scaled)

    QUESTIONNAIRE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        QUESTIONNAIRE_MODEL_PATH,
        compress=3
    )

    joblib.dump(
        scaler,
        QUESTIONNAIRE_SCALER_PATH,
        compress=3
    )

    print("Questionnaire model trained successfully.")
    print(f"Model saved: {QUESTIONNAIRE_MODEL_PATH}")
    print(f"Scaler saved: {QUESTIONNAIRE_SCALER_PATH}")


if __name__ == "__main__":
    train_questionnaire_model()