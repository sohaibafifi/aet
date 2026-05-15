"""Aggregate multi-seed energy_train.json files into a single JSON list.

Each per-seed file may itself contain a list (because train.py appends);
this script flattens, optionally enriches with derived stats, and writes
the merged list to --output.

It also prints summary statistics (median, IQR) for energy_wh and CO2,
which feeds the IQR band in scripts/asymptotic.py / scripts/aet_eval.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from statistics import median


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate per-seed training energy logs")
    p.add_argument("--input", action="append", required=True,
                   help="Glob(s) matching energy_train.json files.")
    p.add_argument("--output", required=True, help="Merged output JSON path.")
    args = p.parse_args()

    paths: list[str] = []
    for pat in args.input:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        sys.stderr.write("error: no input files matched\n")
        return 2

    merged: list[dict] = []
    seen_seeds: set = set()
    for path in paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            sys.stderr.write(f"warn: skipping {path}: {e}\n")
            continue
        if isinstance(data, dict):
            data = [data]
        for entry in data:
            seed = entry.get("seed")
            # Avoid duplicate same-seed runs from a checkpoint resumed in-place
            if seed is not None and seed in seen_seeds:
                # Keep the latest, replace previous
                merged = [m for m in merged if m.get("seed") != seed]
            if seed is not None:
                seen_seeds.add(seed)
            entry["_source"] = path
            merged.append(entry)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    energies = [float(m["energy_wh"]) for m in merged if m.get("energy_wh") is not None]
    co2s = [float(m.get("co2_g_total") or m.get("co2_g_operational") or 0.0)
            for m in merged]
    co2s = [c for c in co2s if c > 0]
    summary = {
        "output": args.output,
        "n_records": len(merged),
        "n_unique_seeds": len(seen_seeds),
        "energy_wh": {
            "median": median(energies) if energies else None,
            "p25": _quantile(energies, 0.25) if energies else None,
            "p75": _quantile(energies, 0.75) if energies else None,
            "min": min(energies) if energies else None,
            "max": max(energies) if energies else None,
        },
        "co2_g": {
            "median": median(co2s) if co2s else None,
            "p25": _quantile(co2s, 0.25) if co2s else None,
            "p75": _quantile(co2s, 0.75) if co2s else None,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
