from algorithm.translator import translator
from algorithm.algorithm import algorithm
import yaml
from pathlib import Path


_CONFIGURATION_DIRECTION = Path(__file__).resolve().parent / "configuration_files"
CONFIG = _CONFIGURATION_DIRECTION / "default_configuration.yaml"
config = yaml.safe_load(CONFIG.open())


def run_algorithm():
    algorithm_functions = translator(config)
    history = algorithm(**algorithm_functions)
    return history


if __name__ == "__main__":
    history = run_algorithm()
