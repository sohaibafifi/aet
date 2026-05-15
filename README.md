# AET: Amortized Efficiency Threshold

Energy-accounting framework for comparing neural and heuristic combinatorial-optimization (CO) solvers under honest fixed-vs-marginal cost accounting.

This repository contains the code and configuration for the experiment described in:

> **An Amortized Efficiency Threshold for Comparing Neural and Heuristic Solvers in Combinatorial Optimization**
> Sohaib Afifi. Univ. Artois, UR 3926, LGI2A, France.
> arXiv:[2605.14624](https://arxiv.org/abs/2605.14624)

The framework defines the deployment-volume threshold `AET` above which a neural solver breaks even with a heuristic baseline in total energy or carbon, under an explicit solution-quality constraint. The companion paper instantiates it on CVRP at `n=50` with the attention-based autoregressive solver of Kool et al. (2019), trained for 100 epochs on 20,000 instances over 5 random seeds, against HGS via PyVRP as the heuristic baseline.

---

## Installation

Requires Python 3.12. Recommended workflow uses [`uv`](https://docs.astral.sh/uv/) for fast dependency resolution; `pip` works equally well.

```bash
# With uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Or with pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

All dependencies (training stack, energy backends, HGS baseline, plotting) are installed by default. The energy backend `pyRAPL` is Linux-only and is skipped automatically on macOS / Windows -- the tracker falls back to NVML + codecarbon + TDP.

---

## Quick start (CVRP n=50)

The full pipeline reproduces every figure in the paper. Expect roughly:
training ~4 h (5 seeds on a single GPU), HGS multi-thread budget sweep
~2.6 h on a 24-core CPU, optional mono single-shot ~16 h, plus a few
minutes of post-processing. Around 7 h end-to-end without the mono
sanity run.

```bash
# (1) Generate CVRP test instances (saved under data/cvrp/).
#     Training and validation samples are drawn fresh by rl4co at setup
#     time and per epoch; only the test set is persisted so the neural
#     solver and the HGS baseline see identical instances.
python scripts/generate_data.py seed=1234

# (2) Train the neural solver, 5 seeds
scripts/train_seeded.sh 5

# (3) Aggregate per-seed training energies
python scripts/aggregate_train_energy.py \
    --input  'logs/cvrp50/seed*/checkpoints/energy_train.json' \
    --output logs/cvrp50/energy_train.json

# (4) Inference batch sweep (B in {1, 32, 128, 512, 1024})
python -m cvrp.test

# (5a) HGS baseline -- realistic deployment regime.
#      Multi-thread, full budget sweep (one solver pass per budget in
#      solver_cfg.runtime_sweep_s).
python -m cvrp.solvers solver_cfg.threads_mode=multi

# (5b) [OPTIONAL] HGS baseline -- pessimistic mono-thread sanity check.
#      Single budget, no sweep. Just one data point to compare against
#      the multi-thread numbers (per-instance energies should differ by
#      only ~10-20% due to idle-power contamination, not by num_procs).
python -m cvrp.solvers \
    solver_cfg.threads_mode=mono \
    solver_cfg.runtime_sweep_s=null \
    solver_cfg.max_runtime_s=60

# (6) Compute AET + sensitivity tables
python scripts/aet_eval.py \
    --train-log     logs/cvrp50/energy_train.json \
    --inference-log 'logs/cvrp50/seed*/checkpoints/energy_inference_*.json' \
    --baseline-log  energy_baseline_pyvrp.json \
    --output-dir    paper/figures \
    --deltas        0,1,2,5,10

# (7) Asymptotic figures (paper Fig. 1-2)
python scripts/asymptotic.py \
    --from-csv     paper/figures/aet_results.csv \
    --output-dir   paper/figures \
    --label        asymptotic

# (8) Sensitivity sweep figures (batch, hardware, delta, HGS budget)
python scripts/asymptotic_sweep.py \
    --from-csv     paper/figures/aet_results.csv \
    --output-dir   paper/figures
```

---

## Energy tracker

The tracker is a context manager (`aet.energy.EnergyTracker`) that dispatches through a fallback chain:

1. **codecarbon** -- reports both Wh and gCO2eq via regional grid intensity (default).
2. **Hardware counters** -- NVML (GPU) + pyRAPL (CPU) for raw Wh on Linux.
3. **TDP fallback** -- `P_TDP x wall_time`, flagged as `backend_used="tdp"`.

```python
from aet.energy import EnergyTracker

with EnergyTracker("train", pue=1.4, hardware_id="nvidia-h100",
                   report_embodied=True, country_iso_code="FRA") as t:
    train_loop()
    t.n_items = 200_000  # for throughput

reading = t.reading.to_dict()  # Wh, gCO2eq (op + embodied), throughput
```

The embodied-carbon table covers 45+ datacenter and consumer SKUs (NVIDIA V100 through GB200, AMD MI250x through MI350x, Intel Xeon, AMD EPYC, Apple Silicon M1 through M5 with Pro / Max / Ultra variants). See `aet/energy/embodied.py`.

---

## Configuration

Single file: `configs/config.yaml`. Everything inlined (env, model, trainer, callbacks, logger, energy, solver_cfg). Override at the command line via Hydra dotted paths:

```bash
python -m cvrp.train \
    trainer.max_epochs=30 \
    model.train_data_size=10_000 \
    seed=42

python -m cvrp.solvers solver_cfg.threads_mode=multi

python -m cvrp.test    energy.batch_sweep='[64, 256]'
```

---

## Reproducibility

All numerical values in the paper are produced by the pipeline above. To reproduce exactly:

- Hardware: a single GPU (logged as `generic-gpu`) for training and inference; multi-threaded HGS baseline on the same node's CPU.
- Software: Python 3.12, PyTorch 2.8.0, Lightning 2.6.1, rl4co 0.6.0, PyVRP 0.10.
- Seeds: 1, 2, 3, 4, 5 (handled by `train_seeded.sh`).
- HGS time budget: 10 s per instance (n=50).

---

## Citing

```bibtex
@article{afifi2026aet,
  author  = {Sohaib Afifi},
  title   = {An Amortized Efficiency Threshold for Comparing Neural
             and Heuristic Solvers in Combinatorial Optimization},
  journal = {arXiv preprint arXiv:2605.14624},
  year    = {2026},
}
```

---

## License

MIT. See [LICENSE](LICENSE).
