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


def custom_influence(satisfaction_levels, custom_values):
    matrix = np.zeros((len(satisfaction_levels), len(satisfaction_levels)))
    matrix
