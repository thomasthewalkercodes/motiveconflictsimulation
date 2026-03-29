from algorithm.translator import translator
from algorithm.algorithm import algorithm
from algorithm.save_results import save_simulation, save_influence_matrix, setup_run
import copy
import itertools
import yaml
from pathlib import Path


# region Comment
# this function goes through my yaml and remembers its path
# when it hits a list (that we later have to vary)
# endregion
def find_axes(d, path=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from find_axes(v, path + (k,))
        elif isinstance(v, list) and not (v and isinstance(v[0], dict)):
            yield path + (k,), v


# region Comment
# this one can get directions and find the right path to the value
# endregion
def set_path(d, path, value):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


_DIR = Path(__file__).resolve().parent / "configuration_files"
SERIES_CONFIG = _DIR / "uni_bi_series_cos.yaml"  # <- PASS YOUR YAML HERE

series = yaml.safe_load(SERIES_CONFIG.open())
# this makes the "all_pairs" readable for the axes list finder
# and makes series out of it.
if series["influence"].get("uni_bi_influence", {}).get("motive_focus") == "all_pairs":
    if series["influence"]["uni_bi_influence"].get("unilateral", True):
        all_pairs = list(itertools.permutations(range(series["n_motives"]), 2))
        # permutation because some functions use the unilateral one
    else:  # here combinations since the function inside influence handles the rest
        all_pairs = list(itertools.combinations(range(series["n_motives"]), 2))
    series["influence"]["uni_bi_influence"]["motive_focus"] = all_pairs

if series["decay"].get("cos_decay", {}).get("motive_focus") == "all_motives":
    series["decay"]["cos_decay"]["motive_focus"] = list(range(series["n_motives"]))

if __name__ == "__main__":
    axes = list(find_axes(series))
    paths = [p for p, _ in axes]
    values = [v for _, v in axes]

    for i, combo in enumerate(itertools.product(*values)):
        config = copy.deepcopy(series)
        tag_parts = [series["tag"]]
        for path, value in zip(paths, combo):
            set_path(config, path, value)
        config["tag"] = f"{series['tag']}_{i:05d}"

        run_dir = setup_run(config)
        for sim in range(config["n_simulations"]):
            history = algorithm(**translator(config))
            save_simulation(history, sim, run_dir)
            save_influence_matrix(history, sim, run_dir)
