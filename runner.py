from algorithm.algorithm import algorithm
from pathlib import Path

_CONFIGURATION_DIRECTION = (
    Path(__file__).resolve().parent.parent / "configuration_files"
)
CONFIG = _CONFIGURATION_DIRECTION / "default_configuration.yaml"


def main(config):
    # region Comment
    # This part here gets the YAML configurations and assigns the functions to the algorithm
    # endregion
    algorithm_functions = {
        "steps": config["steps"],
        "activation_check": getattr(
            activation_check, config["activation_check"]["chosen_activation_check"]
        ),
        "growth": getattr(growth, config["growth"]["chosen_growth"]),
        "influence": getattr(influence, config["influence"]["chosen_influence"]),
        "decay": getattr(decay, config["decay"]["chosen_decay"]),
        "starting_values": getattr(
            starting_values, config["starting_values"]["chosen_starting_values"]
        ),
    }

    run_algorithm = algorithm(**algorithm_functions)
