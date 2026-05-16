from fastapi import FastAPI, HTTPException

from app.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
)

from app.schemas import (
    QuestionnaireRequest,
    QuestionnaireResponse,
    GameRequest,
    GameResponse,
    ADHDRiskRequest,
    ADHDRiskResponse,
    HealthResponse,
)

from app.analyzers import (
    analyze_questionnaire,
    analyze_game,
)

from app.conditions import (
    compute_adhd_risk,
)

from app.scoring import (
    get_risk_level,
)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)


# ==========================
# Health Check
# ==========================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():
    return {
        "status": "ok",
        "service": "NeuroVice AI"
    }


# ==========================
# Questionnaire Analysis
# ==========================

@app.post(
    "/analyze/questionnaire",
    response_model=QuestionnaireResponse
)
def questionnaire_analysis(
    payload: QuestionnaireRequest
):
    try:
        result = analyze_questionnaire(
            payload.answers
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================
# Game Analysis
# ==========================

@app.post(
    "/analyze/game",
    response_model=GameResponse
)
def game_analysis(
    payload: GameRequest
):
    try:
        result = analyze_game(
            reaction_times=payload.reaction_times,
            missed_targets=payload.missed_targets,
            wrong_clicks=payload.wrong_clicks,
            premature_clicks=payload.premature_clicks,
            session_duration=payload.session_duration,
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================
# ADHD Risk Fusion
# ==========================

@app.post(
    "/condition/adhd",
    response_model=ADHDRiskResponse
)
def adhd_risk_analysis(
    payload: ADHDRiskRequest
):
    try:
        result = compute_adhd_risk(
            questionnaire_anomaly=payload.questionnaire_anomaly,
            game_anomaly=payload.game_anomaly,
        )

        result["risk_level"] = get_risk_level(
            result["final_risk_score"]
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )