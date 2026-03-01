import numpy as np
import influence

influence = influence.config["chosen_influence"]


def algorithm(steps, activation_check, growth, influence, decay, satisfaction_levels):
    history = {"step": [], "satisfaction levels": [], "active_motive": []}
    active_motive = None

    for step in range(steps):

        # check if there is an active motive or if something has to get activated
        active_motive = activation_check(satisfaction_levels, active_motive)

        # Apply growth and influence if a motive is active
        if active_motive is not None:
            satisfaction_matrix = growth(satisfaction_matrix, active_motive)
            satisfaction_matrix = influence(satisfaction_matrix, active_motive)

        # Apply decay after growth and influence (does not apply to an active motive)
        satisfaction_matrix = decay(satisfaction_matrix, active_motive)

        # configure the matrix (clip and round values)
        satisfaction_matrix = np.clip(satisfaction_matrix, -1, 1)
        satisfaction_matrix = np.round(satisfaction_matrix, 5)

        # Get current satisfaction levels and label them
        satisfaction_levels = satisfaction_matrix.loc["satisfaction"]

        # saving history
        history["step"].append(step)
        history["satisfaction levels"].append(
            satisfaction_levels.loc["satisfaction"].copy()
        )
        history["active_motive"].append(active_motive)
    return history
