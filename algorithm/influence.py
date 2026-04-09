import numpy as np
import pandas as pd


# region Comment
# This sinus_influence function creates a matrix and applies a sinus wave function
# to each motive (so motives close to each other get high values)
# endregion
def cosinus_influence(satisfaction_levels, amplitude, elevation):
    matrix = np.zeros((len(satisfaction_levels), len(satisfaction_levels)))

    for i in range(len(satisfaction_levels)):
        for j in range(len(satisfaction_levels)):
            if i == j:
                matrix[i, j] = 0
            else:
                distance = min(abs(i - j), len(satisfaction_levels) - abs(i - j))
                angle = distance * (2 * np.pi / len(satisfaction_levels))
                matrix[i, j] = amplitude * np.cos(angle) + elevation

    matrix = (matrix + matrix.T) / 2
    matrix = np.round(matrix, 3)
    return matrix


def uni_bi_influence(
    satisfaction_levels, motive_focus, conflict_strength, unilateral=True
):
    matrix = np.zeros((len(satisfaction_levels), len(satisfaction_levels)))
    matrix[motive_focus[0], motive_focus[1]] = conflict_strength
    if not unilateral:
        matrix[motive_focus[1], motive_focus[0]] = conflict_strength

    return matrix


def custom_influence(satisfaction_levels, conflicts):
    matrix = np.zeros((len(satisfaction_levels), len(satisfaction_levels)))
    for conflict in conflicts:
        i, j = conflict["pair"]
        matrix[i, j] = conflict["strength"]
        if not conflict.get("unilateral", True):
            matrix[j, i] = conflict["strength"]
    return matrix


def sinus_custom_influence(satisfaction_levels, amplitude, elevation, conflicts):
    matrix = cosinus_influence(satisfaction_levels, amplitude, elevation)
    custom_matrix = custom_influence(satisfaction_levels, conflicts)
    combined_matrix = matrix + custom_matrix
    combined_matrix = np.round(combined_matrix, 5)
    return combined_matrix


def circumplex_conflict_influence(
    satisfaction_levels, amplitude, elevation,
    warmth_cold_strength=0.0, dominance_submission_strength=0.0
):
    n = len(satisfaction_levels)
    matrix = cosinus_influence(satisfaction_levels, amplitude, elevation)
    conflict_matrix = np.zeros((n, n))

    # Each motive's loading on warmth and dominance axes
    angles = np.array([i * 2 * np.pi / n for i in range(n)])
    warmth_loading = np.cos(angles)   # positive = warm, negative = cold
    dominance_loading = np.sin(angles)  # positive = dominant, negative = submissive

    if warmth_cold_strength != 0.0:
        warm_weight = np.maximum(0, warmth_loading)   # [1, .71, 0, 0, 0, 0, 0, .71]
        cold_weight = np.maximum(0, -warmth_loading)  # [0, 0, 0, .71, 1, .71, 0, 0]
        # Warm motives influence cold motives and vice versa
        conflict_matrix += warmth_cold_strength * np.outer(warm_weight, cold_weight)
        conflict_matrix += warmth_cold_strength * np.outer(cold_weight, warm_weight)

    if dominance_submission_strength != 0.0:
        dom_weight = np.maximum(0, dominance_loading)   # [0, .71, 1, .71, 0, 0, 0, 0]
        sub_weight = np.maximum(0, -dominance_loading)  # [0, 0, 0, 0, 0, .71, 1, .71]
        # Dominant motives influence submissive motives and vice versa
        conflict_matrix += dominance_submission_strength * np.outer(dom_weight, sub_weight)
        conflict_matrix += dominance_submission_strength * np.outer(sub_weight, dom_weight)

    combined_matrix = matrix + conflict_matrix
    combined_matrix = np.round(combined_matrix, 5)
    return combined_matrix
