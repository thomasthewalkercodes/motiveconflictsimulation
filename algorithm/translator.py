import algorithm.influence as influence_module
import algorithm.activation_check as activation_check_module
import algorithm.growth as growth_module
import algorithm.decay as decay_module
import algorithm.starting_values as starting_values_module
import functools


def translator(config):
    chosen = config["activation_check"]["chosen_activation_check"]
    params = config["activation_check"][chosen]
    activation_check_fn = functools.partial(
        getattr(activation_check_module, chosen), **params
    )
    chosen = config["growth"]["chosen_growth"]
    params = config["growth"][chosen]
    growth_fn = functools.partial(getattr(growth_module, chosen), **params)
    chosen = config["influence"]["chosen_influence"]
    params = config["influence"][chosen]
    influence_fn = functools.partial(getattr(influence_module, chosen), **params)
    chosen = config["decay"]["chosen_decay"]
    params = config["decay"][chosen]
    decay_fn = functools.partial(getattr(decay_module, chosen), **params)
    chosen = config["starting_values"]["chosen_starting_values"]
    params = config["starting_values"][chosen]
    starting_values_fn = functools.partial(
        getattr(starting_values_module, chosen),
        num_motives=config["n_motives"],
        **params
    )
    # region Comment
    # This part here gets the YAML configurations and assigns the functions to
    # the algorithm
    # endregion
    algorithm_functions = {
        "steps": config["steps"],
        "activation_check": activation_check_fn,
        "growth": growth_fn,
        "influence": influence_fn,
        "decay": decay_fn,
        "starting_values": starting_values_fn,
    }
    return algorithm_functions
