from algorithm.translator import translator
from algorithm.algorithm import algorithm
import yaml
from pathlib import Path
import random
import numpy as np
import os
from datetime import datetime
import shutil
import pandas as pd

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
    np.random.seed(config["seed"])  # setting both seeds as they are different
    random.seed(config["seed"])

    run_name = f"{config['tag']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = Path("runs") / run_name
    os.makedirs(run_dir)  # this here creates the folder on disk
    shutil.copy(
        CONFIG, run_dir / f"{config['tag']}.yaml"
    )  # freeze copying Yaml configs to the right folder
    os.makedirs(
        run_dir / "influence_matrices", exist_ok=True
    )  # subfolder for influence matrix for each run

    #### MAIN SIMULATION LOOP #####
    for sim in range(config["n_simulations"]):
        history = algorithm(
            **translator(config)
        )  # this is the main thing, the rest is just to save the results
        df = pd.DataFrame(
            history["satisfaction levels"],
            columns=[
                f"motive_{i+1}" for i in range(config["n_motives"])
            ],  # the +1 is cuz it usually counts from 0 to 7, but we need 1-8 because
            # we are humans (plus the csv in the end is better with a 1 as the first motive and not a 0)
        )
        df.insert(0, "step", history["step"])
        df.insert(
            1, "active_motive", pd.array(history["active_motive"], dtype="Int64") + 1
        )
        df.to_csv(run_dir / f"simulation_{sim}.csv", index=False)

        df = pd.DataFrame(
            history["influence_matrix"],
            columns=[f"motive_{i+1}" for i in range(config["n_motives"])],
            index=[f"motive_{i+1}" for i in range(config["n_motives"])],
        )
        df.to_csv(
            run_dir / "influence_matrices" / f"influence_matrix_{sim}.csv", index=True
        )
