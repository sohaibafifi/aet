#!/usr/bin/env bash
# Multi-seed training launcher for the AET measurement protocol.
#
# Usage:
#   scripts/train_seeded.sh [n_seeds=5] [extra_hydra_args...]
#
# Each seed writes energy_train.json under a per-seed checkpoint directory.
# After all runs complete, call scripts/aggregate_train_energy.py to merge
# the per-seed JSONs into a single file consumable by scripts/aet_eval.py.


N_SEEDS="${1:-5}"
shift || true

EXTRA_ARGS=("$@")

PY="${PY:-python}"

echo "Running ${N_SEEDS} seeds (CVRP n=50, attention solver)"
for seed in $(seq 1 "${N_SEEDS}"); do
    name="cvrp50/seed${seed}"
    echo "==== seed ${seed} (name=${name}) ===="
    "${PY}" -m cvrp.train \
        seed="${seed}" \
        "hydra.run.dir=logs/${name}" \
        "${EXTRA_ARGS[@]}"
done

echo "All seeds done. Aggregate with:"
echo "  ${PY} scripts/aggregate_train_energy.py \\"
echo "     --input 'logs/cvrp50/seed*/checkpoints/energy_train.json' \\"
echo "     --output logs/cvrp50/energy_train.json"
