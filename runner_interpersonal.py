# This is my program runner for the interpersonal addition
from algorithm.translator import translator
from algorithm.algorithm import algorithm
from algorithm.decay import generate.decay
import functools, yaml, numpy as np
from pathlib import Path
from algorithm.save_results import save_influence_matrix, save_simulation, setup_run
from runner_series import find_axes, set_path

_CONFIGURATION_DIRECTION = Path(__file__).resolve().parent / "configuration_files"

#########################
#########################
CONFIG = _CONFIGURATION_DIRECTION / "your_file.yaml" # <- INPUT YOUR DESIRED CONFIG FILE HERE
#########################
#########################

yaml_config = yaml.safe_load(CONFIG.open())

def get_ratios(history):
    active = [a for a in history["active_motive"] if a is not None]
    counts = np.zeros(N_MOTIVES)
    for a in active:
        counts[a] += 1
    total = counts.sum()
    behavior_ratios = counts / total
    return behavior_ratios

def make_decay(behavior_ratios, reference_decay):
    new_rates = generate_decay(behavior_ratios, decay = reference_decay)
    return functools.partioal(new_rates)

# add some yaml stuff that gets lost in translation for each participant
cfg_a = {**cfg["person_a"], "steps": cfg["steps"],
          "active_motive_steps": cfg["active_motive_steps"], "n_motives": N_MOTIVES}
cfg_b = {**cfg["person_b"], "steps": cfg["steps"],
          "active_motive_steps": cfg["active_motive_steps"], "n_motives": N_MOTIVES}

pipu_a = translator(cfg_a)
pipu_b = translator(cfg_b)
#intial decay (we could say starting point) of a (since b already takes the behavior of a immediately)
decay_a = pipu_a["decay"]

if __name__ == "__main__":
    axes = list(find_axes(yaml_config))
    paths = [p for p, _ in axes]
    values = [v for _, v in axes]

    for i, combo in enumerate(itertools.product(*values)):
        config = copy.deepcopy(yaml_config)
        tag_parts = [yaml_config["tag"]]
        for path, value in zip(paths, combo):
            set_path(config, path, value)
        config["tag"] = f"{yaml_config['tag']}_{i:05d}"

        run_dir = setup_run(
            config
        )  # perpares the githash, variables, parameters, tag, folder to save in.

        for sim in range(config["n_simulations"]):
            history = algorithm(**pipu_a, "decay": decay_a) 
            save_simulation(history, sim, run_dir)
            save_influence_matrix(history, sim, run_dir)
            ratios_a = get_ratios(history)
            decay_b = make_decay(ratios_a, pipu_b["decay"])

            history = algor
