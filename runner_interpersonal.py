# This is my program runner for the interpersonal addition
from algorithm.translator import translator
from algorithm.algorithm import algorithm
from algorithm.decay import generate_decay, ratio_decay
import functools, yaml, numpy as np, itertools, copy
from pathlib import Path
from algorithm.save_results import save_influence_matrix, save_simulation, setup_run
from runner_series import find_axes, set_path

_CONFIGURATION_DIRECTION = Path(__file__).resolve().parent / "configuration_files"

#########################
#########################
CONFIG = (
    _CONFIGURATION_DIRECTION / "your_file.yaml"
)  # <- INPUT YOUR DESIRED CONFIG FILE HERE
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
    new_rates = generate_decay(behavior_ratios, decay_rates=reference_decay)

    return functools.partial(new_rates)


if __name__ == "__main__":

    # add some yaml stuff that gets lost in translation for each participant
    cfg_a = {
        **yaml_config["person_a"],
        "steps": yaml_config["steps"],
        "active_motive_steps": yaml_config["active_motive_steps"],
        "n_motives": N_MOTIVES,
    }
    cfg_b = {
        **yaml_config["person_b"],
        "steps": yaml_config["steps"],
        "active_motive_steps": yaml_config["active_motive_steps"],
        "n_motives": N_MOTIVES,
    }

    pipu_a = translator(cfg_a)
    pipu_b = translator(cfg_b)
    # intial decay (we could say starting point) of a (since b already takes the behavior of a immediately)
    decay_a = pipu_a["decay"]

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
            history = algorithm(**{**pipu_a, "decay": decay_a})
            save_simulation(history, sim, run_dir)
            save_influence_matrix(history, sim, run_dir)
            ratios_a = get_ratios(history)
            decay_b = make_decay(ratios_a, pipu_b["decay"])

            history = algorithm(**{**pipu_b, "decay": decay_b})
            save_simulation(history, sim, run_dir)
            save_influence_matrix(history, sim, run_dir)
            ratios_b = get_ratios(history)
            decay_a = make_decay(ratios_b, pipu_a["decay"])
