from algorithm.translator import translator
from algorithm.algorithm import algorithm
import yaml
from pathlib import Path
from algorithm.save_results import save_influence_matrix, save_simulation, setup_run

###################################################################
# PROGRAMM RUNNER #
#################################################################

_CONFIGURATION_DIRECTION = Path(__file__).resolve().parent / "configuration_files"
# write here what configuration file you want to use!
CONFIG = _CONFIGURATION_DIRECTION / "newname.yaml"  # <- HERE
config = yaml.safe_load(CONFIG.open())


# region Comment
# This part runs the algorithm and saves the results and config
# endregion
if __name__ == "__main__":
    run_dir = setup_run(config, CONFIG)
    for sim in range(config["n_simulations"]):
        history = algorithm(
            **translator(config)
        )  # this is the main thing, the rest is just to save the results
        save_simulation(history, sim, run_dir)
        save_influence_matrix(history, sim, run_dir)
