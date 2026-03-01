import numpy as np
import influence

influence = influence.config["chosen_influence"]


def algorithm(steps, activation_check, growth, influence, decay, starting_values):
    history = {"step": [], "satisfaction levels": [], "active_motive": []}
    active_motive = None
    satisfaction_levels = starting_values(config)
    influence_matrix = influence(config)

    for step in range(steps):

        # check if there is an active motive or if something has to get activated
        active_motive = activation_check(satisfaction_levels, active_motive)

        # Apply growth and influence if a motive is active
        if active_motive is not None:
            satisfaction_levels = growth(satisfaction_levels, active_motive)
            satisfaction_levels += influence_matrix.iloc[active_motive]

        # Apply decay after growth and influence (does not apply to an active motive)
        satisfaction_levels = decay(satisfaction_levels, active_motive)

        # configure the matrix (clip and round values)
        satisfaction_levels = np.clip(satisfaction_levels, -1, 1)
        satisfaction_levels = np.round(satisfaction_levels, 5)

        # Get current satisfaction levels and label them
        satisfaction_levels = satisfaction_leves.loc["satisfaction"]

        # saving history
        history["step"].append(step)
        history["satisfaction levels"].append(
            satisfaction_levels.loc["satisfaction"].copy()
        )
        history["active_motive"].append(active_motive)
    return history
