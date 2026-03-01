# imagine a function here that

if active_motive is None or activation_check(active_motive) == True:
    if unsatisfied_motives:
        active_motive = activation_choice(unsatisfied_motives)
    else:
        active_motive = None


unsatisfied_motives = satisfaction_check(satisfaction_levels)  # not needed?


def select_unsatisfied_behavior(satisfaction_levels, unsatisfied_octants):
    if not unsatisfied_octants:
        return None
    # Find the index of the most dissatisfied (lowest satisfaction)
    values = satisfaction_levels[unsatisfied_octants]
    most_dissatisfied_index = np.argmin(values)
    return unsatisfied_octants[most_dissatisfied_index]
