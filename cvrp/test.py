import json
import os
import time

from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch

from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rl4co import utils
from rl4co.data.dataset import FastTdDataset
from rl4co.utils.ops import unbatchify
from tqdm import tqdm

from aet.energy import EnergyTracker


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

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)


@utils.task_wrapper
def run(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # We instantiate the environment separately and then pass it to the model
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    # Note that the RL environment is instantiated inside the model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, env)

    log.info("Instantiating checkpoint callback...")
    checkpoint_callback: Callback = hydra.utils.instantiate(
        cfg.get("callbacks").get("model_checkpoint")
    )

    log.info("Instantiating loggers...")
    loggers: List[Logger] = utils.instantiate_loggers(cfg.get("logger"), model=model)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "callbacks": [checkpoint_callback],
        "logger": loggers,
    }

    if cfg.get("compile", False):
        log.info("Compiling model!")
        model = torch.compile(model)

    log.info("Starting testing!")
    checkpoint_dir = checkpoint_callback.dirpath
    import re

    ckpt_path = cfg.get("ckpt_path") or ""

    if not ckpt_path:
        if not os.path.isdir(checkpoint_dir):
            log.warning(
                f"Checkpoint dir {checkpoint_dir} does not exist. "
                "Pass ckpt_path=... or hydra.run.dir=<seed-dir> to point at a trained run."
            )
            return {}, object_dict

        ckpt_files = [
            f
            for f in os.listdir(checkpoint_dir)
            if re.match(r"^epoch(?:_(\d+))?\.ckpt$", f)
        ]

        if ckpt_files:

            def get_version(filename):
                m = re.match(r"^epoch(?:_(\d+))?\.ckpt$", filename)
                return int(m.group(1)) if m and m.group(1) else 0

            ckpt_file = max(ckpt_files, key=get_version)
            ckpt_path = os.path.join(checkpoint_dir, ckpt_file)

    log.info(f"Using checkpoint path: {ckpt_path}")
    if ckpt_path == "" or not os.path.exists(ckpt_path):
        log.warning("Best ckpt not found! ")
        return {}, object_dict

    # Training checkpoints contain a REINFORCE rollout-baseline copy of the
    # policy under "baseline.baseline.policy.*"; it is not needed for
    # inference. Drop it and load non-strict to tolerate any other
    # training-only keys (e.g. optimizer/baseline buffers).
    raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    inference_state = {
        k: v for k, v in raw_state.items() if not k.startswith("baseline.")
    }
    missing, unexpected = model.load_state_dict(inference_state, strict=False)
    if missing:
        log.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys when loading checkpoint: {unexpected}")
    results = dict()

    model = model.eval().to(device)
    num_augment = getattr(model, "num_augment", cfg.get("model", {}).get("num_augment", 1))
    data_dir = cfg.paths.get("data_dir", "data")

    energy_cfg = cfg.get("energy", {}) or {}
    hardware_id = energy_cfg.get("hardware_id") or _autodetect_hardware_id()
    batch_sweep = list(energy_cfg.get("batch_sweep") or [])
    energy_log_path = os.path.join(
        checkpoint_dir, f"energy_inference_{cfg.get('seed', 'noseed')}.json"
    )
    energy_records: list[dict] = []

    def _inference_loop(loader_):
        local_rewards = torch.tensor([], device=device)
        for batch in tqdm(loader_, total=len(loader_), desc=f"Testing {variant}"):
            with torch.inference_mode():
                with torch.amp.autocast("cuda"):
                    td = env.reset(batch).to(device)
                    out = model.policy(
                        td, env, phase="test", return_actions=True
                    )
                    reward = out["reward"]
                    local_rewards = torch.cat([local_rewards, reward], dim=0)
        return local_rewards

    # Normalize test_file / test_dataloader_names into parallel lists so
    # the loop below works with both scalar and list configurations.
    raw_files = cfg.env.test_file
    if isinstance(raw_files, str) or raw_files is None:
        test_files = [raw_files] if raw_files else []
    else:
        test_files = list(raw_files)
    raw_names = cfg.env.get("test_dataloader_names", None)
    if raw_names is None:
        names = [cfg.env.get("name", "test") for _ in test_files]
    elif isinstance(raw_names, str):
        names = [raw_names]
    else:
        names = list(raw_names)
    if len(names) != len(test_files):
        names = [cfg.env.get("name", "test") for _ in test_files]

    for variant, test_file in zip(names, test_files):
        log.info(f"Testing on {variant} dataset loaded from {test_file}")
        # NOTE: CVRPEnv.load_data normalizes demand via
        #     td["demand"] / td["capacity"][:, None]
        # which assumes external Uchoa-style files where capacity has
        # shape (N,). Our generator (rl4co's CVRPGenerator) stores
        # capacity with shape (N, 1); the extra unsqueeze then broadcasts
        # to (N, num_loc, num_loc) and corrupts the demand tensor. We
        # bypass the override and normalize manually.
        from rl4co.data.utils import load_npz_to_tensordict
        td_data = load_npz_to_tensordict(
            os.path.join(data_dir, test_file)
        ).to(device)
        # Normalize demand by capacity, handling both (N,) and (N, 1)
        # capacity layouts.
        if "capacity" in td_data.keys() and "demand" in td_data.keys():
            cap = td_data["capacity"]
            if cap.ndim == td_data["demand"].ndim:
                # capacity already broadcast-aligned, e.g. (N, 1)
                td_data.set("demand", td_data["demand"] / cap)
            else:
                td_data.set(
                    "demand", td_data["demand"] / cap.unsqueeze(-1)
                )
        # Defensive: persisted tensordicts can come back with a multi-dim
        # batch (e.g. (1, N) instead of (N,)) depending on how they were
        # generated. The dataloader and env._reset both assume a single
        # instance dim, so flatten anything else.
        if len(td_data.batch_size) != 1:
            log.info(
                f"Flattening td_data batch_size {tuple(td_data.batch_size)} -> 1D"
            )
            td_data = td_data.reshape(-1)
        n_instances = td_data.batch_size[0] if td_data.batch_size else len(td_data)
        log.info(f"Loaded {n_instances} test instances, td shape {tuple(td_data.batch_size)}")

        batch_sizes_to_run = batch_sweep if batch_sweep else [cfg.model.test_batch_size]
        all_max_aug_reward = None
        for bsz in batch_sizes_to_run:
            loader = model._dataloader(
                FastTdDataset(td_data), batch_size=int(bsz), shuffle=False
            )
            tracker = EnergyTracker(
                label=f"infer_{variant}_b{bsz}",
                backend=energy_cfg.get("backend", "codecarbon"),
                pue=energy_cfg.get("pue", 1.4),
                hardware_id=hardware_id,
                report_embodied=energy_cfg.get("report_embodied", True),
                lifetime_s=energy_cfg.get("lifetime_s", None),
                grid_intensity_g_per_kwh=energy_cfg.get("grid_intensity_g_per_kwh", 475.0),
                country_iso_code=energy_cfg.get("country_iso_code", None),
                output_dir=None,
            )

            if device == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            elif device == "mps":
                torch.mps.synchronize()
                start = torch.mps.Event(enable_timing=True)
                end = torch.mps.Event(enable_timing=True)
                start.record()
            else:
                start_time = time.process_time()

            with tracker:
                rewards = _inference_loop(loader)
                tracker.n_items = int(n_instances)

            if device == "cuda":
                end.record()
                torch.cuda.synchronize()
                cpu = start.elapsed_time(end) / 1000
            elif device == "mps":
                end.record()
                torch.mps.synchronize()
                cpu = start.elapsed_time(end) / 1000
            else:
                cpu = time.process_time() - start_time

            costs_bks = td_data.get("costs_bks", None)
            avg_gap = None
            if costs_bks is not None:
                mask = costs_bks != float("-inf")
                masked_bks = costs_bks[mask].to(rewards.device)
                masked_reward = rewards[mask]
                denom = masked_bks.abs()
                gap = 100.0 * (-masked_reward - denom) / denom
                avg_gap = gap.mean().cpu().item()

            reading = tracker.reading.to_dict()
            reading.update(
                {
                    "variant": variant,
                    "test_file": test_file,
                    "batch_size": int(bsz),
                    "n_instances": int(n_instances),
                    "device": device,
                    "wall_event_s": cpu,
                    "gap_to_bks": avg_gap,
                    "avg_reward": rewards.mean().cpu().item(),
                    "seed": cfg.get("seed"),
                }
            )
            energy_records.append(reading)
            log.info(
                f"[{variant} b={bsz}] energy={reading['energy_wh']} Wh, "
                f"throughput={reading['throughput_items_per_s']} inst/s, "
                f"gap={avg_gap}"
            )

            if all_max_aug_reward is None or bsz == cfg.model.test_batch_size:
                all_max_aug_reward = rewards

        results[variant] = (
            all_max_aug_reward.mean(dim=0).cpu().item()
            if all_max_aug_reward is not None
            else float("nan")
        )

        for logger in loggers:
            last = energy_records[-1]
            logger.log_metrics(
                {
                    f"test/max_aug_reward/{variant}": float(last["avg_reward"]),
                    f"test/gap_to_bks/{variant}": last.get("gap_to_bks"),
                    f"test/time/{variant}": float(last["wall_event_s"]),
                    f"test/energy_wh/{variant}": last.get("energy_wh") or 0.0,
                    f"test/co2_g/{variant}": last.get("co2_g_total") or 0.0,
                    f"test/throughput/{variant}": last.get("throughput_items_per_s")
                    or 0.0,
                }
            )

    if energy_records:
        try:
            existing = []
            if os.path.exists(energy_log_path):
                with open(energy_log_path, "r") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            existing.extend(energy_records)
            with open(energy_log_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            log.info(f"Inference energy log written to {energy_log_path}")
        except Exception as e:
            log.warning(f"Failed to write inference energy log: {e}")

    log.info(f"Test metrics: {results}")

    return results, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def test(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    metric_dict, _ = run(cfg)

    if not metric_dict:
        log.warning("No metrics returned (likely missing checkpoint). Skipping metric_value.")
        return None
    metric_value = sum([v for v in metric_dict.values()]) / len(metric_dict)

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    test()
