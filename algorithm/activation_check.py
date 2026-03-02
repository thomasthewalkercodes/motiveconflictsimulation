import numpy as np


def linear_check(satisfaction_levels, active_motive):
    if active_motive is not None and satisfaction_levels[active_motive] >= 1:
        active_motive = None
    if active_motive is None:
        if any(satisfaction_levels < 0):
            active_motive = np.argmin(satisfaction_levels)
    return active_motive
