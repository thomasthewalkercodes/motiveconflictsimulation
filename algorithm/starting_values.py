import numpy as np


def random_starting_values(num_motives, low=-1, high=1):
    starting_values = np.random.uniform(low, high, num_motives)
    starting_values = np.round(starting_values, 5)
    return starting_values
