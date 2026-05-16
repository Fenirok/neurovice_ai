from typing import Dict, List
from pydantic import BaseModel, Field, field_validator


# ==========================
# Questionnaire Schemas
# ==========================

class QuestionnaireRequest(BaseModel):
    answers: Dict[str, int] = Field(
        ...,
        description="Questionnaire responses as numeric values"
    )

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, value):
        if not value:
            raise ValueError("Questionnaire answers cannot be empty.")

        for key, answer in value.items():
            if not isinstance(answer, int):
                raise ValueError(f"{key} must be an integer.")

            if answer < 0 or answer > 4:
                raise ValueError(f"{key} must be between 0 and 4.")

        return value


class QuestionnaireResponse(BaseModel):
    anomaly_score: float
    feature_vector: Dict[str, float]


# ==========================
# Game Schemas
# ==========================

class GameRequest(BaseModel):
    reaction_times: List[float]
    missed_targets: int
    wrong_clicks: int
    premature_clicks: int
    session_duration: float

    @field_validator("reaction_times")
    @classmethod
    def validate_reaction_times(cls, value):
        if not value:
            raise ValueError("Reaction times cannot be empty.")

        if len(value) < 5:
            raise ValueError("At least 5 reaction samples required.")

        for rt in value:
            if rt <= 0:
                raise ValueError("Reaction times must be positive.")

        return value

    @field_validator(
        "missed_targets",
        "wrong_clicks",
        "premature_clicks"
    )
    @classmethod
    def validate_non_negative(cls, value):
        if value < 0:
            raise ValueError("Counts cannot be negative.")
        return value

    @field_validator("session_duration")
    @classmethod
    def validate_duration(cls, value):
        if value <= 0:
            raise ValueError("Session duration must be positive.")
        return value


class GameResponse(BaseModel):
    anomaly_score: float
    feature_vector: Dict[str, float]


# ==========================
# ADHD Condition Schemas
# ==========================

class ADHDRiskRequest(BaseModel):
    questionnaire_anomaly: float
    game_anomaly: float

    @field_validator("questionnaire_anomaly", "game_anomaly")
    @classmethod
    def validate_scores(cls, value):
        if value < 0 or value > 1:
            raise ValueError("Scores must be between 0 and 1.")
        return value


class ADHDRiskResponse(BaseModel):
    condition: str
    final_risk_score: float
    risk_level: str


# ==========================
# Health Schema
# ==========================

class HealthResponse(BaseModel):
    status: str
    service: str