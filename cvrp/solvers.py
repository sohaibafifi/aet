import multiprocessing
import sys

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

import json
import os
import re

# ruff: noqa: E402
import time

from typing import Optional

import hydra
import lightning as L

from omegaconf import DictConfig
from rl4co import utils
from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from tensordict.tensordict import TensorDict
from tqdm.auto import tqdm

from aet.energy import EnergyTracker

log = utils.get_pylogger(__name__)

# Fallback per-size HGS time budget (seconds). Overridden by
# solver_cfg.size_to_time_s in config.yaml; only used when that field is
# absent.
DEFAULT_SIZE_TO_TIME = {20: 10, 50: 60, 100: 200}


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def solve(cfg: DictConfig) -> Optional[float]:
    solver_cfg = cfg.get("solver_cfg", {}) or {}
    cpu_total = multiprocessing.cpu_count()
    threads_mode = solver_cfg.get("threads_mode", "default")
    if threads_mode == "mono":
        thread_modes = [("mono", 1)]
    elif threads_mode == "multi":
        thread_modes = [("multi", cpu_total)]
    elif threads_mode == "both":
        thread_modes = [("mono", 1), ("multi", cpu_total)]
    else:
        thread_modes = [("default", cpu_total)]

    solver = solver_cfg.get("name", "pyvrp")
    utils.extras(cfg)

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    output_txt = f"output_{solver}.txt"
    print("Writing output to", output_txt)
    sys.stdout = open(output_txt, "w")

    print("Solver:", solver, "Thread modes:", thread_modes)

    energy_cfg = cfg.get("energy", {}) or {}
    hardware_id = energy_cfg.get("hardware_id", "generic-cpu")
    energy_log_path = solver_cfg.get(
        "energy_log_path", f"energy_baseline_{solver}.json"
    )
    energy_records: list[dict] = []

    # Restrict to test/val files declared by the active env config.
    # Falls back to walking data/ only when the env has no file list.
    env_test_files  = list(cfg.env.get("test_file", []) or [])
    env_val_files   = list(cfg.env.get("val_file",  []) or [])
    declared_files  = env_test_files + env_val_files

    if declared_files:
        # env files are listed relative to data/ (e.g. "all/test_20.npz",
        # "cvrp/test_20.npz"). Pre-existing env layout flattens variants
        # directly under data/, not under data/<env_name>/.
        candidates = [os.path.join("data", f) for f in declared_files]
        data_files = [p for p in candidates if os.path.exists(p)]
        missing = [p for p in candidates if not os.path.exists(p)]
        if missing:
            log.warning(
                f"{len(missing)} env-declared test files missing under data/; "
                "skipped. Run the corresponding generator first."
            )
    else:
        log.warning(
            "env.test_file is empty; falling back to walking data/. This may "
            "include problem classes that the current env does not solve."
        )
        data_files = []
        for root, _dirs, files in os.walk("data"):
            for file in files:
                if re.match(r"test_\d+\.npz$", file) or re.match(r"val_\d+\.npz$", file):
                    data_files.append(os.path.join(root, file))

    data_files = sorted(data_files, key=lambda x: len(x))
    data_files = sorted(data_files, key=lambda x: x.split("/")[-2])
    data_files = sorted(
        data_files,
        key=lambda x: int(
            x.split("/")[-1].split(".")[0].replace("test_", "").replace("val_", "")
        ),
    )

    fixed_runtime = solver_cfg.get("max_runtime_s")  # None | float
    size_to_time = dict(solver_cfg.get("size_to_time_s") or DEFAULT_SIZE_TO_TIME)
    size_to_time = {int(k): float(v) for k, v in size_to_time.items()}

    for file in (pbar := tqdm(data_files, desc="Solving with " + solver)):
        td_test = load_npz_to_tensordict(file)
        demand_key = "demand" if "demand" in td_test.keys() else "demand_linehaul"
        num_problems, size = td_test[demand_key].shape
        if fixed_runtime is not None:
            max_runtime = float(fixed_runtime)
        else:
            max_runtime = size_to_time.get(size - 1, max(size_to_time.values()))

        for mode_name, num_procs in thread_modes:
            sol_suffix = f"_{solver}_{mode_name}" if len(thread_modes) > 1 else f"_{solver}"
            sol_path = file.replace(".npz", f"_sol{sol_suffix}.npz")
            if os.path.exists(sol_path):
                try:
                    sol = load_npz_to_tensordict(sol_path)
                    if (
                        "actions" in sol
                        and "costs" in sol
                        and sol["costs"].shape[0] != num_problems
                    ):
                        log.error(f"num_problems mismatch for {sol_path}, rerunning...")
                    if "actions" in sol and "costs" in sol:
                        log.warning(f"Solution exists for {sol_path}, skipping...")
                        log.info(f"{sol_path} average cost: {-sol['costs'].mean():.3f}")
                        continue
                except RuntimeError as e:
                    log.error(f"Failed to load solution for {sol_path}: {e}, rerunning...")

            print(42 * "=" + f"\nProcessing {file} [{mode_name}, {num_procs} procs]...")
            print(
                f"Estimated time : "
                f"{max_runtime * num_problems / num_procs:.3f} s"
            )
            pbar.set_postfix_str(
                f"{file} [{mode_name}] est: "
                f"{max_runtime * num_problems / num_procs:.3f}s"
            )

            tracker = EnergyTracker(
                label=f"baseline_{solver}_{mode_name}_{os.path.basename(file)}",
                backend=energy_cfg.get("backend", "codecarbon"),
                pue=energy_cfg.get("pue", 1.4),
                hardware_id=hardware_id,
                report_embodied=energy_cfg.get("report_embodied", True),
                lifetime_s=energy_cfg.get("lifetime_s", None),
                grid_intensity_g_per_kwh=energy_cfg.get("grid_intensity_g_per_kwh", 475.0),
                country_iso_code=energy_cfg.get("country_iso_code", None),
                output_dir=None,
            )

            start = time.time()
            with tracker:
                td_local = env.reset(td_test.clone())
                actions_solver, costs_solver = env.solve(
                    td_local,
                    max_runtime=max_runtime,
                    num_procs=num_procs,
                    solver=solver,
                )
                rewards_solver = env.get_reward(td_local.clone(), actions_solver)
                tracker.n_items = int(num_problems)
            total_time = time.time() - start

            reading = tracker.reading.to_dict()
            reading.update(
                {
                    "file": file,
                    "solver": solver,
                    "thread_mode": mode_name,
                    "num_procs": int(num_procs),
                    "num_problems": int(num_problems),
                    "size": int(size),
                    "max_runtime_s": float(max_runtime),
                    "wall_total_s": float(total_time),
                    "avg_cost": float(-rewards_solver.mean()),
                }
            )
            energy_records.append(reading)

            print(f"Time: {total_time:.3f} s | E={reading.get('energy_wh')} Wh | "
                  f"CO2={reading.get('co2_g_total')} g")
            print(f"Average cost: {-rewards_solver.mean():.3f}")

            out = TensorDict(
                {
                    "actions": actions_solver,
                    "costs": costs_solver,
                    "time": total_time,
                    "energy_wh": float(reading.get("energy_wh") or 0.0),
                    "co2_g": float(reading.get("co2_g_total") or 0.0),
                    "num_procs": int(num_procs),
                },
                batch_size=[],
            )
            save_tensordict_to_npz(out, sol_path)

    if energy_records:
        existing = []
        if os.path.exists(energy_log_path):
            try:
                with open(energy_log_path, "r") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except Exception:
                existing = []
        existing.extend(energy_records)
        with open(energy_log_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        log.info(f"Baseline energy log written to {energy_log_path}")


if __name__ == "__main__":
    solve()
