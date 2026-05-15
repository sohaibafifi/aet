import json
import os
import re

from typing import Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch

from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rl4co import utils
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer

from aet.energy import EnergyTracker

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)


def _autodetect_hardware_id() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()
        if "a100" in name:
            return "nvidia-a100"
        if "h100" in name:
            return "nvidia-h100"
        if "v100" in name:
            return "nvidia-v100"
        if "4090" in name:
            return "nvidia-rtx-4090"
        if "3090" in name:
            return "nvidia-rtx-3090"
        return "generic-gpu"
    if torch.backends.mps.is_available():
        return "apple-m2"
    return "generic-cpu"


@utils.task_wrapper
def run(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # We instantiate the environment separately and then pass it to the model
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    # Note that the RL environment is instantiated inside the model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: AttentionModel = hydra.utils.instantiate(cfg.model, env)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"), model)

    log.info("Instantiating trainer...")
    trainer: RL4COTrainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    checkpoint_dir = trainer.checkpoint_callback.dirpath
    if not os.path.exists(checkpoint_dir):
        log.warning(f"Checkpoint directory {checkpoint_dir} does not exist. Creating it.")
        os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_path = cfg.get("ckpt_path") or None
    if ckpt_path is None and cfg.get("auto_resume", False):
        ckpt_files = [
            f for f in os.listdir(checkpoint_dir) if re.match(r"^last(?:-v(\d+))?\.ckpt$", f)
        ]
        if ckpt_files:

            def get_version(filename):
                m = re.match(r"^last(?:-v(\d+))?\.ckpt$", filename)
                return int(m.group(1)) if m.group(1) else 0

            ckpt_file = max(ckpt_files, key=get_version)
            ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
            log.info(f"auto_resume=true: resuming from {ckpt_path}")

    if ckpt_path and not os.path.exists(ckpt_path):
        log.warning(f"ckpt_path {ckpt_path} not found, training from scratch")
        ckpt_path = None

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile", False):
        log.info("Compiling model!")
        model = torch.compile(model)

    train_metrics = {}
    if cfg.get("train"):
        log.info("Starting training!")
        energy_cfg = cfg.get("energy", {}) or {}
        hardware_id = energy_cfg.get("hardware_id") or _autodetect_hardware_id()
        tracker = EnergyTracker(
            label=f"train_{cfg.get('seed', 'noseed')}",
            backend=energy_cfg.get("backend", "codecarbon"),
            pue=energy_cfg.get("pue", 1.4),
            hardware_id=hardware_id,
            report_embodied=energy_cfg.get("report_embodied", True),
            lifetime_s=energy_cfg.get("lifetime_s", None),
            grid_intensity_g_per_kwh=energy_cfg.get("grid_intensity_g_per_kwh", 475.0),
            country_iso_code=energy_cfg.get("country_iso_code", None),
            output_dir=checkpoint_dir,
        )
        with tracker:
            trainer.fit(model=model, ckpt_path=ckpt_path)
            tracker.n_items = int(
                cfg.get("model", {}).get("train_data_size", 0)
            ) * int(cfg.get("trainer", {}).get("max_epochs", 0))

        train_metrics = trainer.callback_metrics

        reading = tracker.reading.to_dict()
        reading["seed"] = cfg.get("seed")
        reading["epochs"] = cfg.get("trainer", {}).get("max_epochs")
        reading["train_data_size"] = cfg.get("model", {}).get("train_data_size")
        energy_path = os.path.join(checkpoint_dir, "energy_train.json")
        # Append-mode for multi-seed cumulative tracking
        existing = []
        if os.path.exists(energy_path):
            try:
                with open(energy_path, "r") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except Exception:
                existing = []
        existing.append(reading)
        with open(energy_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        log.info(
            f"Training energy: {reading.get('energy_wh')} Wh, "
            f"CO2(op)={reading.get('co2_g_operational')} g, "
            f"backend={reading.get('backend_used')}"
        )

    if cfg.get("test"):
        log.info("Starting testing!")
        model.val_batch_size = cfg.get("model").batch_size
        model.test_batch_size = cfg.get("model").batch_size
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path:
            ckpt_path = re.sub(r"^.*(?=/logs/)", os.getcwd(), ckpt_path)

        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def train(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    train()
