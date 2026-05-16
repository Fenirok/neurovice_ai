from pathlib import Path
import pandas as pd
import numpy as np
import json

from app.config import (
    QUESTIONNAIRE_DATA_PATH,
    GAME_DATA_PATH,
)


# ==========================
# Questionnaire Processing
# ==========================

def load_questionnaire_dataset() -> pd.DataFrame:
    """
    Load questionnaire dataset where question_response
    contains JSON question blocks.
    """

    if not QUESTIONNAIRE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Questionnaire dataset not found: {QUESTIONNAIRE_DATA_PATH}"
        )

    df = pd.read_csv(QUESTIONNAIRE_DATA_PATH)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    required_cols = [
        "assessment_id",
        "question_response",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing required questionnaire column: {col}"
            )

    grouped = df.groupby("assessment_id")

    rows = []

    for assessment_id, group in grouped:
        all_responses = []

        for raw_json in group["question_response"]:
            try:
                parsed = json.loads(raw_json)

                values = [
                    float(v)
                    for v in parsed.values()
                    if isinstance(v, (int, float))
                ]

                all_responses.extend(values)

            except Exception:
                continue

        if len(all_responses) < 5:
            continue

        responses = np.array(all_responses, dtype=np.float64)

        avg_response = np.mean(responses)
        response_variance = np.var(responses)
        total_score = np.sum(responses)

        symptom_density = np.sum(responses >= 3) / len(responses)

        response_entropy = (
            len(np.unique(responses)) / len(responses)
        )

        consistency_score = 1.0 / (1.0 + response_variance)

        rows.append({
            "avg_response": avg_response,
            "response_variance": response_variance,
            "total_score": total_score,
            "max_response": np.max(responses),
            "min_response": np.min(responses),
            "symptom_density": symptom_density,
            "response_entropy": response_entropy,
            "consistency_score": consistency_score,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No valid questionnaire samples after processing."
        )

    return result


# ==========================
# Game Processing
# ==========================

def load_game_dataset() -> pd.DataFrame:
    """
    Build game training dataset aligned with runtime inference features.
    """

    if not GAME_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Game dataset not found: {GAME_DATA_PATH}"
        )

    df = pd.read_csv(GAME_DATA_PATH)

    required_cols = [
        "arrow_avg_interval",
        "total_targets_appeared",
        "total_targets_caught",
        "total_clicks",
        "time_left",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing required game column: {col}"
            )

    rows = []

    for _, row in df.iterrows():
        try:
            reaction_mean = float(row["arrow_avg_interval"])

            reaction_std = reaction_mean * 0.15
            coeff_variation = reaction_std / reaction_mean if reaction_mean > 0 else 0

            reaction_skew = 0.0
            reaction_kurtosis = 0.0

            reaction_min = reaction_mean * 0.7
            reaction_max = reaction_mean * 1.3

            total_targets = max(
                float(row["total_targets_appeared"]),
                1.0
            )

            caught = float(row["total_targets_caught"])
            clicks = float(row["total_clicks"])

            missed = total_targets - caught
            wrong = max(clicks - caught, 0)

            omission_error_rate = missed / total_targets
            commission_error_rate = wrong / total_targets
            impulsivity_ratio = wrong / total_targets

            session_duration = max(
                180 - float(row["time_left"]),
                1.0
            )

            click_frequency = clicks / session_duration

            reaction_trend_slope = 0.0

            consistency_score = 1.0 / (1.0 + reaction_std)

            rows.append({
                "reaction_mean": reaction_mean,
                "reaction_std": reaction_std,
                "coeff_variation": coeff_variation,
                "reaction_skew": reaction_skew,
                "reaction_kurtosis": reaction_kurtosis,
                "reaction_min": reaction_min,
                "reaction_max": reaction_max,
                "omission_error_rate": omission_error_rate,
                "commission_error_rate": commission_error_rate,
                "impulsivity_ratio": impulsivity_ratio,
                "click_frequency": click_frequency,
                "reaction_trend_slope": reaction_trend_slope,
                "consistency_score": consistency_score,
            })

        except Exception:
            continue

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No valid game samples after processing."
        )

    return result