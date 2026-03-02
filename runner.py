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


_CONFIGURATION_DIRECTION = Path(__file__).resolve().parent / "configuration_files"
CONFIG = _CONFIGURATION_DIRECTION / "newname.yaml"
config = yaml.safe_load(CONFIG.open())


def run_algorithm():
    algorithm_functions = translator(config)
    history = algorithm(**algorithm_functions)
    return history


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
    )  # freeze copying Yaml configs

    for sim in range(config["n_simulations"]):
        history = algorithm(**translator(config))
        df = pd.DataFrame(
            history["satisfaction levels"],
            columns=[f"motive_{i}" for i in range(config["n_motives"])],
        )
        df.insert(0, "step", history["step"])
        df.insert(1, "active_motive", history["active_motive"])
        df.to_csv(run_dir / f"simulation_{sim}.csv", index=False)
