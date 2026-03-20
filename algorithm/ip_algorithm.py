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
    active_motive_log = []  # records which motive was active each activation
    active_motive = None
    satisfaction_levels = starting_values()
    influence_matrix = influence(satisfaction_levels)
    counter = 0

    for step in range(1, steps):
        active_motive = activation_check(satisfaction_levels, active_motive)

        if active_motive is not None:
            satisfaction_levels = growth(satisfaction_levels, active_motive)
            satisfaction_levels += influence_matrix[active_motive]
            active_motive_log.append(active_motive)
            counter += 1

        satisfaction_levels = decay(satisfaction_levels, active_motive)
        satisfaction_levels = np.clip(satisfaction_levels, -1, 1)
        satisfaction_levels = np.round(satisfaction_levels, 5)

        if counter >= active_motive_steps:
            break

    return {
        "active_motive": active_motive_log,  # list of every active motive activation
        "n_steps": step,  # total steps until algorithm stopped
        "influence_matrix": influence_matrix,
    }
