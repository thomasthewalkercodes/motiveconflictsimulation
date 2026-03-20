import numpy as np


def algorithm(
    steps,
    activation_check,
    growth,
    influence,
    decay,
    starting_values,
    active_motive_steps,
):
    history = {"step": [], "satisfaction levels": [], "active_motive": []}
    active_motive = None
    satisfaction_levels = starting_values()
    influence_matrix = influence(satisfaction_levels)
    history["influence_matrix"] = influence_matrix.copy()
    history["step"].append(0)
    history["satisfaction levels"].append(satisfaction_levels.copy())
    history["active_motive"].append(active_motive)
    counter = 0

    for step in range(1, steps):
        # check if there is an active motive or if something has to get activated
        active_motive = activation_check(satisfaction_levels, active_motive)

        # Apply growth and influence if a motive is active
        if active_motive is not None:
            satisfaction_levels = growth(satisfaction_levels, active_motive)
            satisfaction_levels += influence_matrix[active_motive]
            counter += 1

        # Apply decay after growth and influence (does not apply to an active motive)
        satisfaction_levels = decay(satisfaction_levels, active_motive)

        # configure the matrix (clip and round values)
        satisfaction_levels = np.clip(satisfaction_levels, -1, 1)
        satisfaction_levels = np.round(satisfaction_levels, 5)

        # saving history
        history["step"].append(step)
        history["satisfaction levels"].append(satisfaction_levels.copy())
        history["active_motive"].append(active_motive)
        # checks for how many active motives there are
        if counter >= active_motive_steps:
            break

    return history
