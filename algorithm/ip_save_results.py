import pandas as pd
import yaml
import subprocess
import random
import numpy as np
import os
from datetime import datetime
from pathlib import Path

# This one is used in the runner.py file


def log_run(config, git_hash):
    log_path = Path("master_log.csv")
    # interpersonal configs nest person_a/person_b; use person_a for logging
    agent_cfg = config.get("person_a", config)
    decay_cfg = config.get("decay") or agent_cfg.get("decay", {})
    row = {
        "tag": config["tag"],
        "git_commit": git_hash,
        "steps": config["steps"],
        "active_motive_steps": config["active_motive_steps"],
        "n_simulations": config["n_simulations"],
        "seed": config["seed"],
        "activation_check": agent_cfg["activation_check"]["chosen_activation_check"],
        "activation_check_params": str(
            agent_cfg["activation_check"].get(
                agent_cfg["activation_check"]["chosen_activation_check"], {}
            )
        ),
        "decay": decay_cfg["chosen_decay"],
        "decay_params": str(decay_cfg.get(decay_cfg["chosen_decay"], {})),
        "growth": agent_cfg["growth"]["chosen_growth"],
        "growth_params": str(
            agent_cfg["growth"].get(agent_cfg["growth"]["chosen_growth"], {})
        ),
        "influence": agent_cfg["influence"]["chosen_influence"],
        "influence_params": str(
            agent_cfg["influence"].get(agent_cfg["influence"]["chosen_influence"], {})
        ),
        "starting_values": agent_cfg["starting_values"]["chosen_starting_values"],
        "starting_values_params": str(
            agent_cfg["starting_values"].get(
                agent_cfg["starting_values"]["chosen_starting_values"], {}
            )
        ),
    }

    df = pd.DataFrame([row])
    df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def setup_run(config):
    np.random.seed(config["seed"])  # setting both seeds as they are different
    random.seed(config["seed"])

    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    log_run(config, git_hash)

    run_name = f"{config['tag']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = Path("runs") / run_name
    os.makedirs(run_dir)  # this here creates the folder on disk
    with open(run_dir / f"{config['tag']}.yaml", "w") as f:
        yaml.dump(config, f)
        f.write(f"\ngit_commit: {git_hash}\n")  # save the git hash for reproducibility

    os.makedirs(
        run_dir / "influence_matrices", exist_ok=True
    )  # subfolder for influence matrix for each run
    return run_dir


def save_simulation(history, sim, run_dir):
    df = pd.DataFrame(
        {
            "active_motive": pd.array(history["active_motive"], dtype="Int64") + 1,
        }
    )
    df.to_csv(run_dir / f"simulation_{sim}.csv", index=False)


def save_influence_matrix(history, sim, run_dir):
    df = pd.DataFrame(
        history["influence_matrix"],
        columns=[
            f"motive_{i+1}" for i in range(len(history["satisfaction levels"][0]))
        ],
        index=[f"motive_{i+1}" for i in range(len(history["satisfaction levels"][0]))],
    )
    df.to_csv(
        run_dir / "influence_matrices" / f"influence_matrix_{sim}.csv", index=True
    )
