from typing import Dict, List

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import linregress


# ==========================
# Questionnaire Feature Engineering
# ==========================

def extract_questionnaire_features(
    answers: Dict[str, int]
) -> Dict[str, float]:
    """
    Extract strong questionnaire behavioral features.
    """

    values = np.array(list(answers.values()), dtype=np.float64)

    avg_response = np.mean(values)
    response_variance = np.var(values)
    total_score = np.sum(values)

    max_response = np.max(values)
    min_response = np.min(values)

    symptom_density = np.sum(values >= 3) / len(values)

    response_entropy = (
        len(np.unique(values)) / len(values)
    )

    consistency_score = 1.0 / (1.0 + response_variance)

    return {
        "avg_response": float(avg_response),
        "response_variance": float(response_variance),
        "total_score": float(total_score),
        "max_response": float(max_response),
        "min_response": float(min_response),
        "symptom_density": float(symptom_density),
        "response_entropy": float(response_entropy),
        "consistency_score": float(consistency_score),
    }


# ==========================
# Game Feature Engineering
# ==========================

def extract_game_features(
    reaction_times: List[float],
    missed_targets: int,
    wrong_clicks: int,
    premature_clicks: int,
    session_duration: float,
) -> Dict[str, float]:
    """
    Extract strong behavioral telemetry features.
    """

    rt = np.array(reaction_times, dtype=np.float64)

    total_errors = (
        missed_targets
        + wrong_clicks
        + premature_clicks
    )

    total_events = len(rt) + total_errors

    reaction_mean = np.mean(rt)
    reaction_std = np.std(rt)

    coeff_variation = (
        reaction_std / reaction_mean
        if reaction_mean > 0
        else 0.0
    )

    reaction_skew = skew(rt)
    reaction_kurtosis = kurtosis(rt)

    reaction_min = np.min(rt)
    reaction_max = np.max(rt)

    omission_error_rate = (
        missed_targets / total_events
        if total_events > 0 else 0
    )

    commission_error_rate = (
        wrong_clicks / total_events
        if total_events > 0 else 0
    )

    impulsivity_ratio = (
        premature_clicks / total_events
        if total_events > 0 else 0
    )

    click_frequency = len(rt) / session_duration

    x = np.arange(len(rt))
    slope = linregress(x, rt).slope if len(rt) >= 2 else 0.0

    consistency_score = (
        1.0 / (1.0 + reaction_std)
    )

    return {
        "reaction_mean": float(reaction_mean),
        "reaction_std": float(reaction_std),
        "coeff_variation": float(coeff_variation),
        "reaction_skew": float(reaction_skew),
        "reaction_kurtosis": float(reaction_kurtosis),
        "reaction_min": float(reaction_min),
        "reaction_max": float(reaction_max),
        "omission_error_rate": float(omission_error_rate),
        "commission_error_rate": float(commission_error_rate),
        "impulsivity_ratio": float(impulsivity_ratio),
        "click_frequency": float(click_frequency),
        "reaction_trend_slope": float(slope),
        "consistency_score": float(consistency_score),
    }


# ==========================
# Feature Conversion
# ==========================

def feature_dict_to_array(
    feature_dict: Dict[str, float]
) -> np.ndarray:
    """
    Convert ordered feature dictionary to ML array.
    """

    return np.array(
        list(feature_dict.values()),
        dtype=np.float64
    ).reshape(1, -1)