from app.config import (
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)


def get_risk_level(score: float) -> str:
    """
    Convert numeric score into risk label.
    """

    if score < LOW_RISK_THRESHOLD:
        return "LOW"

    if score < MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"

    return "HIGH"