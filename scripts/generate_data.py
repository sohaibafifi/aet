"""Generate CVRP train / val / test instance files.

Wraps the CVRP generator from rl4co and writes compressed .npz files
under ``data/cvrp/``. Run before ``cvrp.train`` / ``cvrp.test`` /
``cvrp.solvers`` to materialize the instances on disk; rl4co will also
generate on demand the first time a missing file is referenced, but an
explicit one-shot generator makes seeds and counts auditable.

Usage:
    python scripts/generate_data.py                 # uses configs/config.yaml
    python scripts/generate_data.py env.generator_params.num_loc=20
    python scripts/generate_data.py seed=1234 \\
        model.train_data_size=5_000 model.val_data_size=200
"""
from __future__ import annotations

import os

from typing import Optional

import hydra
import lightning as L
import pyrootutils

from omegaconf import DictConfig
from rl4co import utils
from rl4co.data.utils import save_tensordict_to_npz

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def generate(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    num_loc = env.generator.num_loc
    env_name = cfg.env.get("name", "cvrp")
    out_dir = os.path.join("data", env_name)
    os.makedirs(out_dir, exist_ok=True)

    splits = {
        "train": cfg.model.get("train_data_size"),
        "val":   cfg.model.get("val_data_size"),
        "test":  cfg.model.get("test_data_size"),
    }

    for split, size in splits.items():
        if not size:
            continue
        path = os.path.join(out_dir, f"{split}_{num_loc}.npz")
        if os.path.exists(path):
            log.info(f"[skip] {path} already exists ({size} samples expected)")
            continue
        log.info(f"[gen]  {path} ({size} samples)")
        td = env.generator(size)
        save_tensordict_to_npz(td, path, compress=True)

    log.info("Done.")
    return None


if __name__ == "__main__":
    generate()
