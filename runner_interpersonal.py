# This is my program runner for the interpersonal addition
from algorithm.translator import translator
from algorithm.ip_algorithm import algorithm
from algorithm.decay import generate_decay, ratio_decay
import algorithm.decay as decay_module
import functools
import yaml
import numpy as np
import itertools
import copy
from pathlib import Path
from algorithm.ip_save_results import save_influence_matrix, save_simulation, setup_run
from runner_series import find_axes, set_path

_CONFIGURATION_DIRECTION = Path(__file__).resolve().parent / "configuration_files"

#########################
#########################
CONFIG = (
    _CONFIGURATION_DIRECTION / "ip_different_extreme_starts.yaml"
)  # <- INPUT YOUR DESIRED CONFIG FILE HERE
#########################
#########################

yaml_config = yaml.safe_load(CONFIG.open())

# this makes the "all_pairs" readable for the axes list finder
# and makes series out of it.
for person in ["person_a", "person_b"]:
    person_cfg = yaml_config[person]

    uni_bi = person_cfg["influence"].get("uni_bi_influence", {})
    if uni_bi.get("motive_focus") == "all_pairs":
        if uni_bi.get("unilateral", True):
            all_pairs = list(itertools.permutations(range(yaml_config["n_motives"]), 2))
        else:
            all_pairs = list(itertools.combinations(range(yaml_config["n_motives"]), 2))
        person_cfg["influence"]["uni_bi_influence"]["motive_focus"] = all_pairs

if yaml_config["decay"].get("cos_decay", {}).get("motive_focus") == "all_motives":
    yaml_config["decay"]["cos_decay"]["motive_focus"] = list(
        range(yaml_config["n_motives"])
    )


def get_ratios(history, n_motives):
    active = [a for a in history["active_motive"]]
    counts = np.zeros(n_motives)
    for a in active:
        counts[a] += 1
    total = counts.sum()
    behavior_ratios = counts / total
    return behavior_ratios


if __name__ == "__main__":
    axes = list(find_axes(yaml_config))
    paths = [p for p, _ in axes]
    values = [v for _, v in axes]

    for i, combo in enumerate(itertools.product(*values)):
        config = copy.deepcopy(yaml_config)
        for path, value in zip(paths, combo):
            set_path(config, path, value)
        config["tag"] = f"{config['tag']}_{i:05d}"
        n_motives = config["n_motives"]
        n_dialogue = config["n_dialogue"]

        run_dir = setup_run(
            config
        )  # perpares the githash, variables, parameters, tag, folder to save in.

        # add some yaml stuff that gets lost in translation for each participant
        cfg_a = {
            **config["person_a"],
            "steps": config["steps"],
            "active_motive_steps": config["active_motive_steps"],
            "n_motives": n_motives,
            "decay": config["decay"],
        }
        cfg_b = {
            **config["person_b"],
            "steps": config["steps"],
            "active_motive_steps": config["active_motive_steps"],
            "n_motives": n_motives,
            "decay": config["decay"],
        }

        pipu_a = translator(cfg_a)
        pipu_b = translator(cfg_b)
        # intial decay (we could say starting point) of "a"
        # (since b already takes the behavior of a immediately)
        decay_a = functools.partial(
            getattr(decay_module, config["decay"]["chosen_decay"]),
            **config["decay"][config["decay"]["chosen_decay"]],
        )

        for sim in range(config["n_simulations"]):
            for dia_round in range(n_dialogue):
                history = algorithm(**{**pipu_a, "decay": decay_a})
                save_simulation(history, f"a_sim{sim}_round{dia_round}", run_dir)
                if sim == 0 and dia_round == 0:
                    save_influence_matrix(history, "a", run_dir)
                ratios_a = get_ratios(history, n_motives)
                decay_b = functools.partial(
                    ratio_decay,
                    decay_rates=generate_decay(ratios_a, config["total_budget"]),
                )

                history = algorithm(**{**pipu_b, "decay": decay_b})
                save_simulation(history, f"b_sim{sim}_round{dia_round}", run_dir)
                if sim == 0 and dia_round == 0:
                    save_influence_matrix(history, "b", run_dir)
                ratios_b = get_ratios(history, n_motives)
                decay_a = functools.partial(
                    ratio_decay,
                    decay_rates=generate_decay(ratios_b, config["total_budget"]),
                )
