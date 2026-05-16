import sys
from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# Enable app imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.dataset_loader import load_game_dataset
from app.config import (
    GAME_MODEL_PATH,
    GAME_SCALER_PATH,
    GAME_N_ESTIMATORS,
    GAME_CONTAMINATION,
    RANDOM_STATE,
)


def train_game_model():
    """
    Train behavioral anomaly detection model.
    """

    print("Loading game telemetry dataset...")

    df = load_game_dataset()

    print(f"Loaded {len(df)} behavioral samples")

    X = df.values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=GAME_N_ESTIMATORS,
        contamination=GAME_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training behavioral Isolation Forest...")

    model.fit(X_scaled)

    GAME_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        GAME_MODEL_PATH,
        compress=3
    )

    joblib.dump(
        scaler,
        GAME_SCALER_PATH,
        compress=3
    )

    print("Game model trained successfully.")
    print(f"Model saved: {GAME_MODEL_PATH}")
    print(f"Scaler saved: {GAME_SCALER_PATH}")


if __name__ == "__main__":
    train_game_model()