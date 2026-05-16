import numpy as np

from app.preprocessing import (
    extract_questionnaire_features,
    extract_game_features,
    feature_dict_to_array,
)

from app.model_loader import (
    load_model,
    load_scaler,
)

from app.config import (
    QUESTIONNAIRE_MODEL_PATH,
    QUESTIONNAIRE_SCALER_PATH,
    GAME_MODEL_PATH,
    GAME_SCALER_PATH,
)


# ==========================
# Score Calibration
# ==========================

def calibrate_anomaly_score(raw_score: float) -> float:
    """
    Convert Isolation Forest decision score
    into stable 0-1 anomaly probability-like score.

    Lower IF score = more anomalous.
    """

    calibrated = 1.0 / (1.0 + np.exp(5.0 * raw_score))

    return float(np.clip(calibrated, 0.0, 1.0))


# ==========================
# Questionnaire Analyzer
# ==========================

def analyze_questionnaire(answers: dict):
    """
    Full questionnaire anomaly pipeline.
    """

    features = extract_questionnaire_features(answers)

    feature_array = feature_dict_to_array(features)

    scaler = load_scaler(QUESTIONNAIRE_SCALER_PATH)
    model = load_model(QUESTIONNAIRE_MODEL_PATH)

    scaled = scaler.transform(feature_array)

    raw_score = model.decision_function(scaled)[0]

    anomaly_score = calibrate_anomaly_score(raw_score)

    return {
        "anomaly_score": anomaly_score,
        "feature_vector": features,
    }


# ==========================
# Game Analyzer
# ==========================

def analyze_game(
    reaction_times,
    missed_targets,
    wrong_clicks,
    premature_clicks,
    session_duration,
):
    """
    Full behavioral anomaly pipeline.
    """

    features = extract_game_features(
        reaction_times=reaction_times,
        missed_targets=missed_targets,
        wrong_clicks=wrong_clicks,
        premature_clicks=premature_clicks,
        session_duration=session_duration,
    )

    feature_array = feature_dict_to_array(features)

    scaler = load_scaler(GAME_SCALER_PATH)
    model = load_model(GAME_MODEL_PATH)

    scaled = scaler.transform(feature_array)

    raw_score = model.decision_function(scaled)[0]

    anomaly_score = calibrate_anomaly_score(raw_score)

    return {
        "anomaly_score": anomaly_score,
        "feature_vector": features,
    }