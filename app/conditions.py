import numpy as np


def clamp(value: float) -> float:
    """
    Keep score between 0 and 1.
    """
    return float(np.clip(value, 0.0, 1.0))


def confidence_weighted_fusion(
    questionnaire_score: float,
    game_score: float
) -> float:
    """
    Stronger fusion logic.

    Higher anomaly confidence contributes more.
    Behavioral telemetry is weighted slightly higher.
    """

    questionnaire_confidence = max(
        questionnaire_score,
        0.15
    )

    game_confidence = max(
        game_score,
        0.20
    )

    questionnaire_weight = 0.45 * questionnaire_confidence
    game_weight = 0.55 * game_confidence

    total_weight = (
        questionnaire_weight + game_weight
    )

    fused = (
        questionnaire_score * questionnaire_weight
        + game_score * game_weight
    ) / total_weight

    return clamp(fused)


def compute_adhd_risk(
    questionnaire_anomaly: float,
    game_anomaly: float
):
    """
    ADHD risk interpretation engine.
    """

    fused_score = confidence_weighted_fusion(
        questionnaire_score=questionnaire_anomaly,
        game_score=game_anomaly
    )

    impulsivity_boost = 0.0

    if game_anomaly > 0.80:
        impulsivity_boost += 0.05

    if questionnaire_anomaly > 0.75:
        impulsivity_boost += 0.03

    final_score = clamp(
        fused_score + impulsivity_boost
    )

    return {
        "condition": "ADHD",
        "final_risk_score": final_score
    }