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
    return pd.DataFrame(
        matrix,
        columns=[f"motive_{i+1}" for i in range(len(satisfaction_levels))],
        index=[f"motive_{i+1}" for i in range(len(satisfaction_levels))],
    )
