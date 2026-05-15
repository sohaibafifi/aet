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
SMOKE_BATCHES  = [1, 32, 128, 512, 1024]
SMOKE_HARDWARE = ["laptop-cpu", "server-cpu", "consumer-gpu", "datacenter-gpu"]
SMOKE_DELTAS   = [0.0, 1.0, 2.0, 5.0, 10.0]
SMOKE_SEEDS    = list(range(5))
SMOKE_BUDGETS  = [1.0, 5.0, 10.0, 30.0, 60.0, 120.0]   # HGS per-instance budgets (s)

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


def smoke_e_meta_per_inst_wh(
    threads_mode: str,
    hardware: str,
    budget_s: float = 10.0,
) -> float:
    """Per-instance metaheuristic energy at the given HGS budget.

    Wall time per instance ~= budget. Mono uses one core; multi uses
    multiple cores with sub-linear scaling. Per-instance energy is
    roughly the same in both modes (each instance runs on one logical
    worker); idle-power contamination makes mono ~10-20% higher.
    """
    p_w = HARDWARE_POWER_W[hardware]
    if threads_mode == "mono":
        # 1 active core for budget seconds + idle overhead.
        return p_w * budget_s / 3600.0 * 1.15
    elif threads_mode == "multi":
        # Same energy per instance, slightly lower with better idle amortization.
        return p_w * budget_s / 3600.0
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
    """Cartesian product of (batch, hardware, delta, seed, budget) with
    derived energies. The budget axis is the HGS per-instance time budget
    (seconds); per-instance metaheuristic energy scales linearly in it.
    """
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
                    for budget in SMOKE_BUDGETS:
                        e_mono  = smoke_e_meta_per_inst_wh("mono",  "server-cpu", budget)
                        e_multi = smoke_e_meta_per_inst_wh("multi", "server-cpu", budget)
                        rows.append({
                            "hardware":   hw,
                            "batch":      batch,
                            "delta_pct":  delta,
                            "seed":       seed,
                            "budget_s":   budget,
                            "E_train_wh": e_train,
                            "E_NN_wh_per_inst":         e_nn,
                            "E_meta_wh_per_inst_mono":  e_mono,
                            "E_meta_wh_per_inst_multi": e_multi,
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


def plot_by_budget(rows: list[dict], out_path: str) -> dict:
    """One NN curve per (constant); one metaheuristic line per HGS budget.

    The neural side is invariant to the HGS budget; what shifts is the
    metaheuristic per-instance energy (which scales linearly with the
    HGS budget at fixed thread count). We therefore plot a single NN
    curve at the throughput plateau and overlay one metaheuristic line
    per budget value, marking the crossover for each.
    """
    hw = _pick_default_hardware(rows)
    budgets = sorted({
        r["budget_s"] for r in rows
        if isinstance(r.get("budget_s"), float) and r["budget_s"] == r["budget_s"]
    })
    if len(budgets) < 2:
        # Single-budget run: nothing to sweep; emit a stub figure.
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5,
                "Budget sweep requires solver_cfg.runtime_sweep_s\n"
                "with at least 2 values.",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"figure": out_path, "crossovers": {}}

    batches_seen = sorted({r["batch"] for r in rows})
    plateau_batch = batches_seen[-1] if batches_seen else max(SMOKE_BATCHES)
    held = [r for r in rows
            if r["hardware"] == hw and r["batch"] == plateau_batch
            and r["delta_pct"] == 1.0]
    if not held:
        held = [r for r in rows if r["hardware"] == hw and r["batch"] == plateau_batch]

    # NN: one median curve.
    e_train_samples = np.array([r["E_train_wh"] for r in held])
    e_nn = float(np.median([r["E_NN_wh_per_inst"] for r in held]))
    nn_curve = CurveSpec(
        label=f"NN (B={plateau_batch}, E_NN={e_nn:.2e} Wh/inst)",
        color="#1f77b4",
        e_train_samples=e_train_samples,
        e_nn_per_inst=e_nn,
        feasible_share=float(np.mean([r["feasible"] for r in held])) if held else 1.0,
    )

    N = _n_grid()
    fig, ax = plt.subplots(figsize=(9, 6))
    e_train_med = float(np.median(e_train_samples))
    nn_med = e_train_med + e_nn * N
    ax.plot(N, nn_med, color=nn_curve.color, linewidth=2.0, label=nn_curve.label)

    # Meta lines: one per budget (multi-thread baseline).
    colors = _viridis_colors(len(budgets))
    annotations: dict[str, float] = {}
    for budget, color in zip(budgets, colors):
        sub = [r for r in rows
               if r["budget_s"] == budget and r["hardware"] == hw
               and r["batch"] == plateau_batch]
        if not sub:
            continue
        e_meta = float(np.median([r["E_meta_wh_per_inst_multi"] for r in sub]))
        ax.plot(N, e_meta * N, linestyle="--", linewidth=1.4, color=color,
                label=f"HGS multi, t={budget:g}s (E={e_meta:.2e} Wh/inst)")
        denom = e_meta - e_nn
        if denom > 0:
            n_star = e_train_med / denom
            if 1.0 <= n_star <= 1e10:
                ax.plot([n_star], [e_meta * n_star], marker="o",
                        color=color, markersize=7, markeredgecolor="black",
                        markeredgewidth=0.6, zorder=5)
                annotations[f"t={budget:g}s"] = float(n_star)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (deployed instances)")
    ax.set_ylabel("Cumulative energy (Wh)")
    ax.set_title(
        f"AET sensitivity: HGS time budget  "
        f"(hardware = {hw}, batch = {plateau_batch}, multi-thread baseline)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure": out_path, "crossovers": annotations}


# ----------------------------------------------------------------------
# Combined 2x2 sweep figure (batch / hardware / delta / budget overlay)
# ----------------------------------------------------------------------

def _curves_by_batch(rows):
    hw = _pick_default_hardware(rows)
    held = [r for r in rows if r["hardware"] == hw and r["delta_pct"] == 1.0]
    by = _group_by(held, "batch")
    keys = sorted(by.keys())
    colors = _viridis_colors(len(keys))
    curves = []
    for k, col in zip(keys, colors):
        sub = by[k]
        e_nn = float(np.median([r["E_NN_wh_per_inst"] for r in sub]))
        curves.append(CurveSpec(
            label=f"B = {k:>4d}",
            color=col,
            e_train_samples=np.array([r["E_train_wh"] for r in sub]),
            e_nn_per_inst=e_nn,
            feasible_share=float(np.mean([r["feasible"] for r in sub])),
        ))
    return curves, held, f"batch size  (hardware = {hw}, $\\delta$=1%)"


def _curves_by_hardware(rows):
    plateau_batch = max({r["batch"] for r in rows}) if rows else max(SMOKE_BATCHES)
    held = [r for r in rows if r["batch"] == plateau_batch and r["delta_pct"] == 1.0]
    by = _group_by(held, "hardware")
    seen = set(by.keys())
    order = [h for h in SMOKE_HARDWARE if h in seen] + sorted(seen - set(SMOKE_HARDWARE))
    colors = _viridis_colors(len(order))
    curves = []
    for k, col in zip(order, colors):
        sub = by[k]
        e_nn = float(np.median([r["E_NN_wh_per_inst"] for r in sub]))
        curves.append(CurveSpec(
            label=f"{k}",
            color=col,
            e_train_samples=np.array([r["E_train_wh"] for r in sub]),
            e_nn_per_inst=e_nn,
            feasible_share=float(np.mean([r["feasible"] for r in sub])),
        ))
    return curves, held, f"hardware  (B={plateau_batch}, $\\delta$=1%)"


def _curves_by_delta(rows):
    hw = _pick_default_hardware(rows)
    plateau_batch = max({r["batch"] for r in rows}) if rows else max(SMOKE_BATCHES)
    held = [r for r in rows if r["hardware"] == hw and r["batch"] == plateau_batch]
    by = _group_by(held, "delta_pct")
    keys = sorted(by.keys())
    colors = _viridis_colors(len(keys))
    curves = []
    for k, col in zip(keys, colors):
        sub = by[k]
        e_nn = float(np.median([r["E_NN_wh_per_inst"] for r in sub]))
        feas = float(np.mean([r["feasible"] for r in sub]))
        marker = "" if feas >= 0.5 else " [infeas.]"
        curves.append(CurveSpec(
            label=f"$\\delta$ = {k:g}%{marker}",
            color=col,
            e_train_samples=np.array([r["E_train_wh"] for r in sub]),
            e_nn_per_inst=e_nn,
            feasible_share=feas,
        ))
    return curves, held, f"quality tolerance  (hardware = {hw}, B={plateau_batch})"


def plot_combined(rows: list[dict], out_path: str) -> dict:
    """2x2 subplot grid: batch / hardware / delta / budget axes.

    Each subplot mirrors a ``plot_by_*`` function but shares one figure
    for at-a-glance comparison. Useful as the main paper sensitivity
    figure when a single composite is desired.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    summary: dict[str, dict] = {}

    def _meta_lines(held):
        if not held:
            return float("nan"), float("nan")
        return (
            float(np.median([r["E_meta_wh_per_inst_mono"]  for r in held])),
            float(np.median([r["E_meta_wh_per_inst_multi"] for r in held])),
        )

    curves, held, t = _curves_by_batch(rows)
    e_mo, e_mu = _meta_lines(held)
    summary["batch"] = _plot_family(axes[0, 0], curves, e_mo, e_mu,
                                    title=f"AET sensitivity: {t}")

    curves, held, t = _curves_by_hardware(rows)
    e_mo, e_mu = _meta_lines(held)
    summary["hardware"] = _plot_family(axes[0, 1], curves, e_mo, e_mu,
                                       title=f"AET sensitivity: {t}")

    curves, held, t = _curves_by_delta(rows)
    e_mo, e_mu = _meta_lines(held)
    summary["delta"] = _plot_family(axes[1, 0], curves, e_mo, e_mu,
                                    title=f"AET sensitivity: {t}")

    # (1, 1) HGS-budget panel: one NN line + one meta line per budget.
    ax = axes[1, 1]
    hw = _pick_default_hardware(rows)
    plateau_batch = max({r["batch"] for r in rows}) if rows else max(SMOKE_BATCHES)
    budgets = sorted({
        r["budget_s"] for r in rows
        if isinstance(r.get("budget_s"), float) and r["budget_s"] == r["budget_s"]
    })
    if len(budgets) >= 2:
        held = [r for r in rows
                if r["hardware"] == hw and r["batch"] == plateau_batch
                and r["delta_pct"] == 1.0]
        if not held:
            held = [r for r in rows if r["hardware"] == hw and r["batch"] == plateau_batch]
        e_train_samples = np.array([r["E_train_wh"] for r in held])
        e_train_med = float(np.median(e_train_samples)) if held else 0.0
        e_nn = float(np.median([r["E_NN_wh_per_inst"] for r in held])) if held else 0.0
        N = _n_grid()
        ax.plot(N, e_train_med + e_nn * N, color="#1f77b4", linewidth=2.0,
                label=f"NN (B={plateau_batch})")
        colors = _viridis_colors(len(budgets))
        budget_annot: dict[str, float] = {}
        for budget, col in zip(budgets, colors):
            sub = [r for r in rows
                   if r["budget_s"] == budget and r["hardware"] == hw
                   and r["batch"] == plateau_batch]
            if not sub:
                continue
            e_meta = float(np.median([r["E_meta_wh_per_inst_multi"] for r in sub]))
            ax.plot(N, e_meta * N, linestyle="--", linewidth=1.4, color=col,
                    label=f"HGS multi, t={budget:g}s")
            denom = e_meta - e_nn
            if denom > 0:
                n_star = e_train_med / denom
                if 1.0 <= n_star <= 1e10:
                    ax.plot([n_star], [e_meta * n_star], marker="o", color=col,
                            markersize=7, markeredgecolor="black",
                            markeredgewidth=0.6, zorder=5)
                    budget_annot[f"t={budget:g}s"] = float(n_star)
        summary["budget"] = budget_annot
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (deployed instances)")
        ax.set_ylabel("Cumulative energy (Wh)")
        ax.set_title(
            f"AET sensitivity: HGS time budget  "
            f"(hardware = {hw}, B={plateau_batch}, multi-thread)"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    else:
        ax.text(0.5, 0.5,
                "Budget sweep requires runtime_sweep_s\nwith at least 2 values.",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        summary["budget"] = {}

    fig.suptitle("AET sensitivity surface (all axes)", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure": out_path, "crossovers": summary}


# ----------------------------------------------------------------------
# Envelope (fill_between) sweep figure
# ----------------------------------------------------------------------

def plot_envelope(rows: list[dict], out_path: str) -> dict:
    """Single-plot envelope of cumulative energies across all sweep axes.

    For each N on the deployment grid, compute the min / median / max
    of cumulative neural energy across all (batch, hardware, delta,
    seed) combinations, and the min / median / max of cumulative
    metaheuristic energy across (thread_mode, budget) combinations.
    Shaded bands show full spread; solid lines show medians. The
    region where the NN envelope sits below the HGS envelope marks
    the deployment regime where the network wins for every
    practitioner choice; the crossover band marks the AET interval.
    """
    if not rows:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, "No data.", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"figure": out_path, "crossovers": {}}

    N = _n_grid()

    # NN cumulative: stack per-row curves
    nn_cum_rows = []
    for r in rows:
        if r["feasible"] is False:
            continue
        nn_cum_rows.append(r["E_train_wh"] + r["E_NN_wh_per_inst"] * N)
    if not nn_cum_rows:
        nn_cum_rows = [r["E_train_wh"] + r["E_NN_wh_per_inst"] * N for r in rows]
    nn_cum = np.vstack(nn_cum_rows)

    nn_min = nn_cum.min(axis=0)
    nn_max = nn_cum.max(axis=0)
    nn_med = np.median(nn_cum, axis=0)

    # HGS cumulative: take min/max over BOTH thread modes and budgets
    meta_vals = []
    for r in rows:
        meta_vals.append(r["E_meta_wh_per_inst_mono"])
        meta_vals.append(r["E_meta_wh_per_inst_multi"])
    meta_arr = np.array([v for v in meta_vals if np.isfinite(v) and v > 0])
    if meta_arr.size == 0:
        meta_arr = np.array([1e-3])
    e_meta_min = float(meta_arr.min())
    e_meta_max = float(meta_arr.max())
    e_meta_med = float(np.median(meta_arr))

    meta_cum_min = e_meta_min * N
    meta_cum_max = e_meta_max * N
    meta_cum_med = e_meta_med * N

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # NN envelope (blue)
    ax.fill_between(N, nn_min, nn_max, color="#1f77b4", alpha=0.20,
                    label="NN envelope (across batch / hardware / $\\delta$ / seed)")
    ax.plot(N, nn_med, color="#1f77b4", linewidth=2.2,
            label=f"NN median (E_NN $\\approx$ {np.median([r['E_NN_wh_per_inst'] for r in rows]):.2e} Wh/inst)")

    # HGS envelope (red)
    ax.fill_between(N, meta_cum_min, meta_cum_max, color="#d62728", alpha=0.20,
                    label="HGS envelope (across thread mode / budget)")
    ax.plot(N, meta_cum_med, color="#d62728", linewidth=2.2, linestyle="--",
            label=f"HGS median (E_meta $\\approx$ {e_meta_med:.2e} Wh/inst)")

    # Mark median crossover
    e_nn_med = float(np.median([r["E_NN_wh_per_inst"] for r in rows]))
    e_train_med = float(np.median([r["E_train_wh"] for r in rows]))
    denom = e_meta_med - e_nn_med
    crossover = float("nan")
    if denom > 0:
        crossover = e_train_med / denom
        if 1.0 <= crossover <= 1e10:
            ax.axvline(crossover, linestyle=":", color="black", alpha=0.5)
            ax.annotate(
                f"AET median $\\approx$ {crossover:,.0f}",
                xy=(crossover, e_meta_med * crossover),
                xytext=(crossover * 2.0, e_meta_med * crossover * 0.3),
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.5),
                fontsize=10,
            )

    # Crossover band: range of AET across env (min/max E_meta)
    aet_min = aet_max = float("nan")
    denom_max = e_meta_max - e_nn_med
    denom_min = e_meta_min - e_nn_med
    if denom_max > 0 and denom_min > 0:
        aet_min = e_train_med / denom_max
        aet_max = e_train_med / denom_min
        if all(1.0 <= v <= 1e10 for v in [aet_min, aet_max]):
            ax.axvspan(aet_min, aet_max, color="gray", alpha=0.15,
                       label=f"AET band [{aet_min:,.0f}, {aet_max:,.0f}]")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (deployed instances)")
    ax.set_ylabel("Cumulative energy (Wh)")
    ax.set_title("AET envelope: NN vs HGS across the full sensitivity surface")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure": out_path,
        "crossover_median": crossover,
        "aet_band_min": aet_min,
        "aet_band_max": aet_max,
        "e_nn_median": e_nn_med,
        "e_meta_min":  e_meta_min,
        "e_meta_med":  e_meta_med,
        "e_meta_max":  e_meta_max,
    }


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
    has_budget = "baseline_max_runtime_s" in df.columns
    key_cols = ["hardware_id", "batch_size", "delta_pct", "seed"]
    if has_budget:
        key_cols.append("baseline_max_runtime_s")
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
        if has_budget:
            hardware_id, batch_size, delta_pct, seed, budget = keys
        else:
            hardware_id, batch_size, delta_pct, seed = keys
            budget = float("nan")
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
            "budget_s":   float(budget) if budget == budget else float("nan"),  # NaN-safe
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
    summary["figures"]["budget"]   = plot_by_budget(rows,
                                                    os.path.join(args.output_dir, "aet_by_budget.png"))
    summary["figures"]["combined"] = plot_combined(rows,
                                                   os.path.join(args.output_dir, "aet_sensitivity_overview.png"))
    summary["figures"]["envelope"] = plot_envelope(rows,
                                                   os.path.join(args.output_dir, "aet_envelope.png"))

    summary_path = os.path.join(args.output_dir, "aet_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
