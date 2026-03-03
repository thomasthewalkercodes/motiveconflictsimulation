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
        elif isinstance(v, list):
            yield path + (k,), v


# region Comment
# this one can get directions and find the right path to the value
# endregion
def set_path(d, path, value):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


_DIR = Path(__file__).resolve().parent / "configuration_files"
SERIES_CONFIG = _DIR / "example.yaml"  # <- PASS YOUR YAML HERE

series = yaml.safe_load(SERIES_CONFIG.open())

if __name__ == "__main__":
    axes = list(find_axes(series))
    paths = [p for p, _ in axes]
    values = [v for _, v in axes]

    for combo in itertools.product(*values):
        config = copy.deepcopy(series)
        tag_parts = [series["tag_prefix"]]
        for path, value in zip(paths, combo):
            set_path(config, path, value)
            tag_parts.append(f"{path[-1]}_{value}")
        config["tag"] = "_".join(str(p) for p in tag_parts)

        run_dir = setup_run(config)
        for sim in range(config["n_simulations"]):
            history = algorithm(**translator(config))
            save_simulation(history, sim, run_dir)
            save_influence_matrix(history, sim, run_dir)
