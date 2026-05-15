"""Generate the CVRP test instance file.

The training loop samples fresh instances every epoch via
``env.dataset(train_data_size, phase="train")``; the validation loop
likewise re-samples fresh val instances each fit() setup. Only the
test set is persisted on disk, so the same instances are used by both
the neural solver (``cvrp.test``) and the HGS baseline
(``cvrp.solvers``) when reporting AET numbers.

Output file is written under ``data/<env_name>/`` and skipped if
already present.

Usage:
    python scripts/generate_data.py                          # use config defaults
    python scripts/generate_data.py env.generator_params.num_loc=100
    python scripts/generate_data.py seed=1234 model.test_data_size=500
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

    # Training and validation data are sampled fresh by rl4co at setup
    # / per epoch; only the test set is persisted to keep the neural
    # solver and the HGS baseline running on the exact same instances.
    size = cfg.model.get("test_data_size")
    if size:
        path = os.path.join(out_dir, f"test_{num_loc}.npz")
        if os.path.exists(path):
            log.info(f"[skip] {path} already exists ({size} samples expected)")
        else:
            log.info(f"[gen]  {path} ({size} samples)")
            td = env.generator(size)
            save_tensordict_to_npz(td, path, compress=True)

    log.info("Done.")
    return None


if __name__ == "__main__":
    generate()
