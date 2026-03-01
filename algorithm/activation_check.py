import numpy as np


def linear_check(satisfaction_levels, active_motive):
    if active_motive is None and any(satisfaction_levels < 0):
        active_motive = np.argmin(satisfaction_levels)
    else:
        if satisfaction_levels[active_motive] >= 1:
            active_motive = None

    return active_motive
