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
import numpy as np
import torch

from omegaconf import DictConfig
from rl4co import utils
from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from tensordict.tensordict import TensorDict
from tqdm.auto import tqdm

from aet.energy import EnergyTracker

log = utils.get_pylogger(__name__)

# ----------------------------------------------------------------------
# Pure-CVRP PyVRP wrapper. We call PyVRP directly,
# using a process pool for batched parallelism across instances.
#
# Integer scaling conventions:
#   - coordinates are sampled in [0, 1]; we scale by 1e6 so PyVRP's
#     internal integer distance matrix has enough resolution.
#   - demands and vehicle capacity are integers in the CVRP convention
#     (capacity = 40 for n=50, demands sampled in 1..10). Generator
#     stores demand as (demand_int / capacity); we recover the integer
#     via round(demand_norm * capacity).
# ----------------------------------------------------------------------
PYVRP_COORD_SCALE = 1_000_000  # 1e6


def _pyvrp_solve_one(args):
    """Solve a single CVRP instance with PyVRP. Returns (action, cost).

    `action` is the giant-tour representation expected by rl4co's
    get_reward (a list of node indices starting with the depot, with
    a 0 inserted between routes).
    `cost` is the unscaled tour length (sum of Euclidean distances).
    """
    coords_np, depot_np, demand_int_np, capacity_int, max_runtime = args

    # Lazy import inside the worker so multiprocessing on `spawn` /
    # `forkserver` start methods does not pickle pyvrp at the parent.
    from pyvrp import Client, Depot, ProblemData, VehicleType
    from pyvrp import solve as pyvrp_solve
    from pyvrp.stop import MaxRuntime

    num_clients = demand_int_np.shape[0]

    # Build scaled-integer distance matrix (depot is node 0).
    full_coords = np.concatenate([depot_np[None, :], coords_np], axis=0)
    scaled = np.round(full_coords * PYVRP_COORD_SCALE).astype(np.int64)
    diff = scaled[:, None, :] - scaled[None, :, :]
    dist_int = np.round(np.sqrt((diff.astype(np.float64) ** 2).sum(-1))).astype(np.int64)

    depot = Depot(x=int(scaled[0, 0]), y=int(scaled[0, 1]))
    clients = [
        Client(
            x=int(scaled[i + 1, 0]),
            y=int(scaled[i + 1, 1]),
            delivery=[int(demand_int_np[i])],
        )
        for i in range(num_clients)
    ]
    vehicle = VehicleType(
        num_available=num_clients,
        capacity=[int(capacity_int)],
    )
    data = ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle],
        distance_matrices=[dist_int],
        duration_matrices=[np.zeros_like(dist_int)],
    )

    result = pyvrp_solve(data, stop=MaxRuntime(float(max_runtime)))
    sol = result.best
    # Giant-tour action: concat routes separated by 0 (depot).
    action: list[int] = []
    for route in sol.routes():
        action.extend(route.visits())
        action.append(0)
    # Unscale cost back to original coordinate units.
    cost = float(result.cost()) / PYVRP_COORD_SCALE
    return action, cost


def _pyvrp_solve_batch(td_test, max_runtime: float, num_procs: int):
    """Solve every instance in a CVRP TensorDict with PyVRP.

    Returns:
        actions: padded LongTensor of shape (N, T_max) padded with 0.
        costs:   FloatTensor of shape (N,) (tour lengths, lower is better).
    """
    locs = td_test["locs"].cpu().numpy()  # (N, K, 2)
    depots = td_test["depot"].cpu().numpy()  # (N, 2)
    demand_norm = td_test["demand"].cpu().numpy()  # (N, K) normalized
    capacity = td_test["capacity"].cpu().numpy()  # (N, 1) or (N,)
    capacity = capacity.reshape(capacity.shape[0], -1)[:, 0]  # (N,)

    # Recover integer demand: saved as int / capacity.
    demand_int = np.round(demand_norm * capacity[:, None]).astype(np.int64)
    capacity_int = np.round(capacity).astype(np.int64)

    n_instances = locs.shape[0]
    work = [
        (locs[i], depots[i], demand_int[i], int(capacity_int[i]), max_runtime)
        for i in range(n_instances)
    ]

    if num_procs > 1:
        with multiprocessing.Pool(processes=int(num_procs)) as pool:
            results = list(
                tqdm(
                    pool.imap(_pyvrp_solve_one, work),
                    total=n_instances,
                    desc=f"PyVRP procs={num_procs} t={max_runtime}s",
                    leave=False,
                )
            )
    else:
        results = [
            _pyvrp_solve_one(w)
            for w in tqdm(work, desc=f"PyVRP mono t={max_runtime}s", leave=False)
        ]

    # Pad actions to a fixed length tensor.
    actions = [r[0] for r in results]
    costs = torch.tensor([r[1] for r in results], dtype=torch.float32)
    t_max = max(len(a) for a in actions) if actions else 0
    actions_pad = torch.zeros((n_instances, t_max), dtype=torch.long)
    for i, a in enumerate(actions):
        actions_pad[i, : len(a)] = torch.tensor(a, dtype=torch.long)
    return actions_pad, costs

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
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        return list(x)

    env_test_files = _as_list(cfg.env.get("test_file"))
    env_val_files  = _as_list(cfg.env.get("val_file"))
    declared_files = env_test_files + env_val_files

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
    runtime_sweep = solver_cfg.get("runtime_sweep_s")  # None | list[float]

    for file in (pbar := tqdm(data_files, desc="Solving with " + solver)):
        td_test = load_npz_to_tensordict(file)
        demand_key = "demand" if "demand" in td_test.keys() else "demand_linehaul"
        num_problems, size = td_test[demand_key].shape

        # Resolve the list of per-instance budgets to sweep.
        if runtime_sweep:
            budgets = [float(b) for b in runtime_sweep]
        elif fixed_runtime is not None:
            budgets = [float(fixed_runtime)]
        else:
            budgets = [float(size_to_time.get(size - 1, max(size_to_time.values())))]

        for mode_name, num_procs in thread_modes:
            for budget in budgets:
                budget_tag = f"_t{int(round(budget))}" if len(budgets) > 1 else ""
                sol_suffix = (
                    f"_{solver}_{mode_name}{budget_tag}"
                    if len(thread_modes) > 1 or len(budgets) > 1
                    else f"_{solver}"
                )
                sol_path = file.replace(".npz", f"_sol{sol_suffix}.npz")
                if os.path.exists(sol_path):
                    try:
                        sol = load_npz_to_tensordict(sol_path)
                        if (
                            "actions" in sol
                            and "costs" in sol
                            and sol["costs"].shape[0] != num_problems
                        ):
                            log.error(
                                f"num_problems mismatch for {sol_path}, rerunning..."
                            )
                        if "actions" in sol and "costs" in sol:
                            log.warning(f"Solution exists for {sol_path}, skipping...")
                            log.info(
                                f"{sol_path} average cost: {-sol['costs'].mean():.3f}"
                            )
                            continue
                    except RuntimeError as e:
                        log.error(
                            f"Failed to load solution for {sol_path}: {e}, rerunning..."
                        )

                print(
                    42 * "="
                    + f"\nProcessing {file} [{mode_name}, {num_procs} procs, "
                    f"t={budget:g}s]..."
                )
                print(
                    f"Estimated time : "
                    f"{budget * num_problems / num_procs:.3f} s"
                )
                pbar.set_postfix_str(
                    f"{file} [{mode_name} t={budget:g}s] est: "
                    f"{budget * num_problems / num_procs:.3f}s"
                )

                tracker = EnergyTracker(
                    label=(
                        f"baseline_{solver}_{mode_name}_t{int(round(budget))}_"
                        f"{os.path.basename(file)}"
                    ),
                    backend=energy_cfg.get("backend", "codecarbon"),
                    pue=energy_cfg.get("pue", 1.4),
                    hardware_id=hardware_id,
                    report_embodied=energy_cfg.get("report_embodied", True),
                    lifetime_s=energy_cfg.get("lifetime_s", None),
                    grid_intensity_g_per_kwh=energy_cfg.get(
                        "grid_intensity_g_per_kwh", 475.0
                    ),
                    country_iso_code=energy_cfg.get("country_iso_code", None),
                    output_dir=None,
                )

                start = time.time()
                with tracker:
                    if solver == "pyvrp":
                        actions_solver, costs_solver = _pyvrp_solve_batch(
                            td_test, max_runtime=budget, num_procs=num_procs
                        )
                        # cost is tour length; reward convention in this
                        # repo is negative cost (consistent with rl4co).
                        rewards_solver = -costs_solver
                    else:
                        raise NotImplementedError(
                            f"solver={solver!r} not supported; only 'pyvrp' is "
                            f"wired up. Add a wrapper analogous to "
                            f"_pyvrp_solve_batch."
                        )
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
                        "max_runtime_s": float(budget),
                        "wall_total_s": float(total_time),
                        "avg_cost": float(-rewards_solver.mean()),
                    }
                )
                energy_records.append(reading)

                print(
                    f"Time: {total_time:.3f} s | "
                    f"E={reading.get('energy_wh')} Wh | "
                    f"CO2={reading.get('co2_g_total')} g | "
                    f"budget={budget:g}s"
                )
                print(f"Average cost: {-rewards_solver.mean():.3f}")

                out = TensorDict(
                    {
                        "actions":     actions_solver,
                        "costs":       costs_solver,
                        "time":        total_time,
                        "energy_wh":   float(reading.get("energy_wh") or 0.0),
                        "co2_g":       float(reading.get("co2_g_total") or 0.0),
                        "num_procs":   int(num_procs),
                        "max_runtime_s": float(budget),
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
