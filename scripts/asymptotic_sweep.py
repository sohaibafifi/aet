"""Multi-config asymptotic sweep plotter for the AET framework.

For each fixed sensitivity axis (batch, hardware, delta), plot one
cumulative-energy curve per axis value over deployed instance count N,
with an inter-seed IQR band; overlay two metaheuristic baseline lines
(mono- and multi-thread) and annotate crossover points.

Two data sources are supported:
  (a) --from-csv  : aet_results.csv produced by scripts/aet_eval.py
  (b) --smoke     : generate plausible synthetic data anchored to the
                    values reported in docs/aet/figures/asymptotic_summary.json,
                    so the figure set can be assembled while the empirical
                    pipeline is still running.

Output PNGs are written to --output-dir (default docs/aet/figures/) under
the names aet_by_batch.png, aet_by_hardware.png, aet_by_delta.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dataclasses import dataclass

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"matplotlib + numpy required: {e}\n")
    raise


# ----------------------------------------------------------------------
# Smoke-data generator
# ----------------------------------------------------------------------

# Anchor values match docs/aet/figures/asymptotic_summary.json:
#   E_train = 42120 Wh on a datacenter GPU
#   E_NN per inst at batch 1024 = 4.07e-6 Wh
#   E_meta per inst mono = 0.125 Wh
SMOKE_BATCHES = [1, 32, 128, 512, 1024]
SMOKE_HARDWARE = ["laptop-cpu", "server-cpu", "consumer-gpu", "datacenter-gpu"]
SMOKE_DELTAS = [0.0, 1.0, 2.0, 5.0, 10.0]
SMOKE_SEEDS = list(range(5))

# Per-hardware power draws (W) and embodied-carbon shorthand.
HARDWARE_POWER_W = {
    "laptop-cpu":     30.0,
    "server-cpu":     45.0,
    "consumer-gpu":   200.0,
    "datacenter-gpu": 300.0,
}
# Training is GPU-only; CPUs do not train neural solvers in our setting.
HARDWARE_IS_TRAINABLE = {
    "laptop-cpu":     False,
    "server-cpu":     False,
    "consumer-gpu":   True,
    "datacenter-gpu": True,
}
# Median training energy per GPU tier (Wh), with a ~20% IQR.
HARDWARE_E_TRAIN_WH = {
    "consumer-gpu":   62000.0,  # consumer card slower => more wall-clock
    "datacenter-gpu": 42120.0,  # anchor value
}


def throughput_inst_per_s(batch: int) -> float:
    """Model GPU throughput as a saturating function of batch size."""
    t_batch_s = 0.005 + 0.000044 * batch  # 5 ms + 44 µs per item, saturating
    return batch / t_batch_s


def smoke_e_nn_per_inst_wh(batch: int, hardware: str) -> float:
    """Per-instance NN inference energy at the given batch on the given GPU."""
    if not HARDWARE_IS_TRAINABLE[hardware]:
        return float("nan")
    p_w = HARDWARE_POWER_W[hardware]
    tau = throughput_inst_per_s(batch)
    return p_w / (3600.0 * tau)


def smoke_e_meta_per_inst_wh(threads_mode: str, hardware: str) -> float:
    """Per-instance metaheuristic energy. Threads scale time but not power."""
    p_w = HARDWARE_POWER_W[hardware]
    t_meta_s = 10.0  # PyVRP wall time per instance on a single thread
    if threads_mode == "mono":
        return p_w * t_meta_s / 3600.0
    elif threads_mode == "multi":
        # ~8 threads with sub-linear scaling; speedup ~5x
        return p_w * (t_meta_s / 5.0) / 3600.0
    else:
        raise ValueError(f"unknown threads_mode: {threads_mode}")


def smoke_gap_pct(seed: int) -> float:
    """Synthetic gap of the neural solver, as a function of seed.

    Gap distribution chosen so that tight tolerances (delta=0) yield
    infeasibility on most seeds while loose tolerances (delta>=2) are
    feasible uniformly. Produces feasible/infeasible zones in the
    by-delta plot.
    """
    rng = np.random.default_rng(seed)
    base = 0.8 + 0.4 * rng.standard_normal()  # mean 0.8%, sd 0.4%
    return max(0.0, base)


def generate_smoke_rows() -> list[dict]:
    """Cartesian product of (batch, hardware, delta, seed) with derived energies."""
    rows: list[dict] = []
    for hw in SMOKE_HARDWARE:
        if not HARDWARE_IS_TRAINABLE[hw]:
            continue
        e_train_med = HARDWARE_E_TRAIN_WH[hw]
        for batch in SMOKE_BATCHES:
            e_nn = smoke_e_nn_per_inst_wh(batch, hw)
            for delta in SMOKE_DELTAS:
                for seed in SMOKE_SEEDS:
                    rng = np.random.default_rng(seed + hash((hw, batch)) % 1000)
                    e_train = float(e_train_med * (1.0 + 0.20 * rng.standard_normal()))
                    gap_nn = smoke_gap_pct(seed + 11 * batch)
                    gap_base = 0.0  # pyvrp ~ optimal
                    feasible = gap_nn <= gap_base + delta
                    rows.append({
                        "hardware":   hw,
                        "batch":      batch,
                        "delta_pct":  delta,
                        "seed":       seed,
                        "E_train_wh": e_train,
                        "E_NN_wh_per_inst":   e_nn,
                        "E_meta_wh_per_inst_mono":  smoke_e_meta_per_inst_wh("mono",  "server-cpu"),
                        "E_meta_wh_per_inst_multi": smoke_e_meta_per_inst_wh("multi", "server-cpu"),
                        "gap_nn_pct":   gap_nn,
                        "gap_base_pct": gap_base,
                        "feasible":     feasible,
                    })
    return rows


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CurveSpec:
    label: str
    color: str
    e_train_samples: np.ndarray  # over seeds
    e_nn_per_inst:   float
    feasible_share:  float       # 0..1 share of seeds passing feasibility


def _n_grid() -> np.ndarray:
    return np.logspace(0, 9, 100)


def _plot_family(
    ax,
    curves: list[CurveSpec],
    e_meta_mono: float,
    e_meta_multi: float,
    title: str,
) -> dict:
    N = _n_grid()
    annotations: dict[str, float] = {}

    # Metaheuristic lines (straight in log-log with slope 1).
    ax.plot(N, e_meta_mono * N,  linestyle="--", color="#d62728", linewidth=1.6,
            label=f"Metaheuristic, mono-thread  ({e_meta_mono:.2e} Wh/inst)")
    ax.plot(N, e_meta_multi * N, linestyle=":",  color="#d62728", linewidth=1.6,
            label=f"Metaheuristic, multi-thread ({e_meta_multi:.2e} Wh/inst)")

    # NN families (one per axis value).
    for c in curves:
        e_train_med = float(np.median(c.e_train_samples))
        e_train_q25 = float(np.percentile(c.e_train_samples, 25))
        e_train_q75 = float(np.percentile(c.e_train_samples, 75))

        nn_med = e_train_med + c.e_nn_per_inst * N
        nn_lo  = e_train_q25 + c.e_nn_per_inst * N
        nn_hi  = e_train_q75 + c.e_nn_per_inst * N

        style = "-" if c.feasible_share >= 0.5 else "--"
        alpha_band = 0.18 if c.feasible_share >= 0.5 else 0.06
        ax.plot(N, nn_med, color=c.color, linewidth=2.0, linestyle=style,
                label=c.label)
        ax.fill_between(N, nn_lo, nn_hi, color=c.color, alpha=alpha_band)

        # Crossover with the multi-thread metaheuristic line (the fair one).
        denom = e_meta_multi - c.e_nn_per_inst
        if denom > 0 and c.feasible_share >= 0.5:
            n_star = e_train_med / denom
            if 1.0 <= n_star <= 1e10:
                ax.plot([n_star], [e_meta_multi * n_star], marker="o",
                        color=c.color, markersize=7, markeredgecolor="black",
                        markeredgewidth=0.6, zorder=5)
                annotations[c.label] = float(n_star)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (deployed instances)")
    ax.set_ylabel("Cumulative energy (Wh)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    return annotations


def _group_by(rows: list[dict], axis: str) -> dict:
    out: dict[object, list[dict]] = {}
    for r in rows:
        out.setdefault(r[axis], []).append(r)
    return out


def _viridis_colors(n: int) -> list[str]:
    cmap = matplotlib.colormaps.get_cmap("viridis")
    if n == 1:
        return [mcolors.to_hex(cmap(0.5))]
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def _pick_default_hardware(rows: list[dict]) -> str:
    """Pick the hardware id to hold fixed in by-batch / by-delta plots.

    Prefer datacenter GPU when present (smoke convention); otherwise fall
    back to the most common id in the data (real runs typically log a
    single hardware_id, e.g. "generic-gpu" or the auto-detected SKU).
    """
    if any(r["hardware"] == "datacenter-gpu" for r in rows):
        return "datacenter-gpu"
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["hardware"]] = counts.get(r["hardware"], 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""


def plot_by_batch(rows: list[dict], out_path: str) -> dict:
    hw = _pick_default_hardware(rows)
    held = [r for r in rows
            if r["hardware"] == hw and r["delta_pct"] == 1.0]
    by_b = _group_by(held, "batch")
    batches = sorted(by_b.keys())
    colors = _viridis_colors(len(batches))

    curves: list[CurveSpec] = []
    for b, color in zip(batches, colors):
        sub = by_b[b]
        e_train_samples = np.array([r["E_train_wh"] for r in sub])
        e_nn = float(np.median([r["E_NN_wh_per_inst"] for r in sub]))
        feas = float(np.mean([r["feasible"] for r in sub]))
        curves.append(CurveSpec(
            label=f"B = {b:>4d}  (E_NN={e_nn:.2e} Wh/inst)",
            color=color,
            e_train_samples=e_train_samples,
            e_nn_per_inst=e_nn,
            feasible_share=feas,
        ))

    fig, ax = plt.subplots(figsize=(9, 6))
    e_meta_mono  = float(np.median([r["E_meta_wh_per_inst_mono"]  for r in held]))
    e_meta_multi = float(np.median([r["E_meta_wh_per_inst_multi"] for r in held]))
    crosses = _plot_family(
        ax, curves, e_meta_mono, e_meta_multi,
        title=f"AET sensitivity: inference batch size  "
              f"(hardware = {hw}, $\\delta$ = 1%)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure": out_path, "crossovers": crosses}


def plot_by_hardware(rows: list[dict], out_path: str) -> dict:
    batches_seen = sorted({r["batch"] for r in rows})
    plateau_batch = batches_seen[-1] if batches_seen else max(SMOKE_BATCHES)
    held = [r for r in rows
            if r["batch"] == plateau_batch and r["delta_pct"] == 1.0]
    by_h = _group_by(held, "hardware")
    # Prefer canonical ordering, append unknown hardwares afterwards.
    seen = set(by_h.keys())
    hardware_order = [h for h in SMOKE_HARDWARE if h in seen] + \
                     sorted(seen - set(SMOKE_HARDWARE))
    colors = _viridis_colors(len(hardware_order))

    curves: list[CurveSpec] = []
    for h, color in zip(hardware_order, colors):
        sub = by_h[h]
        e_train_samples = np.array([r["E_train_wh"] for r in sub])
        e_nn = float(np.median([r["E_NN_wh_per_inst"] for r in sub]))
        feas = float(np.mean([r["feasible"] for r in sub]))
        curves.append(CurveSpec(
            label=f"{h}  (E_NN={e_nn:.2e} Wh/inst)",
            color=color,
            e_train_samples=e_train_samples,
            e_nn_per_inst=e_nn,
            feasible_share=feas,
        ))

    fig, ax = plt.subplots(figsize=(9, 6))
    e_meta_mono  = float(np.median([r["E_meta_wh_per_inst_mono"]  for r in held]))
    e_meta_multi = float(np.median([r["E_meta_wh_per_inst_multi"] for r in held]))
    crosses = _plot_family(
        ax, curves, e_meta_mono, e_meta_multi,
        title=f"AET sensitivity: training hardware  "
              f"(batch = {plateau_batch}, $\\delta$ = 1%)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure": out_path, "crossovers": crosses}


def plot_by_delta(rows: list[dict], out_path: str) -> dict:
    hw = _pick_default_hardware(rows)
    batches_seen = sorted({r["batch"] for r in rows})
    plateau_batch = batches_seen[-1] if batches_seen else max(SMOKE_BATCHES)
    held = [r for r in rows
            if r["hardware"] == hw and r["batch"] == plateau_batch]
    by_d = _group_by(held, "delta_pct")
    deltas = sorted(by_d.keys())
    colors = _viridis_colors(len(deltas))

    curves: list[CurveSpec] = []
    for d, color in zip(deltas, colors):
        sub = by_d[d]
        e_train_samples = np.array([r["E_train_wh"] for r in sub])
        e_nn = float(np.median([r["E_NN_wh_per_inst"] for r in sub]))
        feas = float(np.mean([r["feasible"] for r in sub]))
        marker = "" if feas >= 0.5 else " [infeasible]"
        curves.append(CurveSpec(
            label=f"$\\delta$ = {d:g}%{marker}",
            color=color,
            e_train_samples=e_train_samples,
            e_nn_per_inst=e_nn,
            feasible_share=feas,
        ))

    fig, ax = plt.subplots(figsize=(9, 6))
    e_meta_mono  = float(np.median([r["E_meta_wh_per_inst_mono"]  for r in held]))
    e_meta_multi = float(np.median([r["E_meta_wh_per_inst_multi"] for r in held]))
    crosses = _plot_family(
        ax, curves, e_meta_mono, e_meta_multi,
        title=f"AET sensitivity: quality tolerance $\\delta$  "
              f"(hardware = {hw}, batch = {plateau_batch})",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure": out_path, "crossovers": crosses}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _csv_to_rows(df) -> list[dict]:
    """Adapt aet_results.csv produced by aet_eval.py to the row schema
    expected by the plotting helpers.

    Real CSV columns of interest:
      variant, size, batch_size, threads_mode (mono|multi), num_procs,
      delta_pct, nn_gap_pct, baseline_gap_pct, feasible, hardware_id,
      seed, E_train_wh_median, E_NN_wh_per_inst, E_meta_wh_per_inst, ...

    The sweep plots treat each (hardware, batch, delta, seed) as a
    single point that needs both mono and multi metaheuristic costs.
    We pivot the threads_mode dimension into two columns
    (E_meta_wh_per_inst_mono, E_meta_wh_per_inst_multi).
    """
    key_cols = ["hardware_id", "batch_size", "delta_pct", "seed"]
    required = {"hardware_id", "batch_size", "delta_pct", "seed",
                "threads_mode", "E_meta_wh_per_inst",
                "E_train_wh_median", "E_NN_wh_per_inst", "feasible"}
    missing = required - set(df.columns)
    if missing:
        sys.stderr.write(
            f"warn: CSV missing columns {missing}; sweep figures may degrade\n"
        )

    rows: list[dict] = []
    for keys, sub in df.groupby(key_cols):
        hardware_id, batch_size, delta_pct, seed = keys
        sub_mono  = sub[sub["threads_mode"] == "mono"]
        sub_multi = sub[sub["threads_mode"] == "multi"]
        if sub_mono.empty and sub_multi.empty:
            continue
        meta_mono  = float(sub_mono["E_meta_wh_per_inst"].median())  if not sub_mono.empty  else float("nan")
        meta_multi = float(sub_multi["E_meta_wh_per_inst"].median()) if not sub_multi.empty else meta_mono
        if sub_mono.empty:
            meta_mono = meta_multi
        any_row = sub.iloc[0]
        rows.append({
            "hardware":   str(any_row.get("hardware_id", hardware_id)),
            "batch":      int(batch_size),
            "delta_pct":  float(delta_pct),
            "seed":       int(seed),
            "E_train_wh": float(any_row["E_train_wh_median"]),
            "E_NN_wh_per_inst":         float(any_row["E_NN_wh_per_inst"]),
            "E_meta_wh_per_inst_mono":  meta_mono,
            "E_meta_wh_per_inst_multi": meta_multi,
            "gap_nn_pct":   float(any_row.get("nn_gap_pct", 0.0)),
            "gap_base_pct": float(any_row.get("baseline_gap_pct", 0.0)),
            "feasible":     bool(any_row.get("feasible", True)),
            "variant":      str(any_row.get("variant", "")),
            "size":         int(any_row.get("size", -1)),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="AET multi-config sweep plotter")
    p.add_argument("--from-csv", default=None,
                   help="aet_results.csv produced by aet_eval.py")
    p.add_argument("--smoke", action="store_true",
                   help="Generate plausible synthetic data instead of reading CSV.")
    p.add_argument("--output-dir", default="docs/aet/figures")
    args = p.parse_args()

    if not args.smoke and not args.from_csv:
        sys.stderr.write("error: provide either --smoke or --from-csv\n")
        return 2

    if args.from_csv:
        try:
            import pandas as pd
        except Exception as e:
            sys.stderr.write(f"pandas required for --from-csv: {e}\n")
            return 2
        df = pd.read_csv(args.from_csv)
        rows = _csv_to_rows(df)
    else:
        rows = generate_smoke_rows()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "n_rows": len(rows),
        "source": "smoke" if args.smoke else args.from_csv,
        "figures": {},
    }
    summary["figures"]["batch"]    = plot_by_batch(rows,
                                                   os.path.join(args.output_dir, "aet_by_batch.png"))
    summary["figures"]["hardware"] = plot_by_hardware(rows,
                                                      os.path.join(args.output_dir, "aet_by_hardware.png"))
    summary["figures"]["delta"]    = plot_by_delta(rows,
                                                   os.path.join(args.output_dir, "aet_by_delta.png"))

    summary_path = os.path.join(args.output_dir, "aet_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
