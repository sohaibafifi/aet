"""Amortized Efficiency Threshold (AET) evaluation.

Inputs
------
- --train-log: one or more JSON files written by train.py (energy_train.json).
  Each file is a list of dicts (one per seed) with keys: energy_wh,
  co2_g_operational, co2_g_total, duration_s, seed, hardware_id, ...
- --inference-log: one or more JSON files written by test.py
  (energy_inference_<seed>.json). Each entry has: variant, batch_size,
  n_instances, energy_wh, energy_wh_per_item, throughput_items_per_s,
  co2_g_operational, co2_g_total, gap_to_bks, seed, hardware_id.
- --baseline-log: one or more JSON files written by solvers.py
  (energy_baseline_<solver>.json). Each entry: file, solver, thread_mode,
  num_procs, num_problems, size, energy_wh, energy_wh_per_item, co2_g_total,
  avg_cost, baseline_gap (optional; default 0).

For each combination of (variant, size, batch_size, thread_mode, delta) we
compute:

    feasible = gap_NN <= baseline_gap + delta
    AET_E = E_train_aggregate / max(E_meta_per_inst - E_NN_per_inst, eps)
    AET_C = C_train_aggregate / max(C_meta_per_inst - C_NN_per_inst, eps)

When `feasible` is False or denominator <= 0, AET is reported as inf.

Outputs
-------
- aet_results.csv (tidy table)
- Plots:
    aet_by_delta.png         - AET vs delta (per batch_size)
    aet_by_batch.png         - AET vs batch_size (per delta)
    aet_by_size.png          - AET vs problem size (with IQR multi-seed)
    aet_threads_bars.png     - mono vs multi-thread comparison
    asymptotic_overview.png  - cumulative energy curves vs N (log-log)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

from dataclasses import dataclass, field
from typing import Iterable, Optional

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"pandas required: {e}\n")
    raise

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"matplotlib required: {e}\n")
    raise


EPS = 1e-12


@dataclass
class TrainAggregate:
    energy_wh_median: float
    energy_wh_p25: float
    energy_wh_p75: float
    co2_g_median: float
    co2_g_p25: float
    co2_g_p75: float
    n_seeds: int
    hardware_id: Optional[str]
    raw: list[dict] = field(default_factory=list)


def _load_json_list(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return list(data)


def _load_many(paths: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for pat in paths:
        for path in sorted(glob.glob(pat)) or [pat]:
            if not os.path.exists(path):
                continue
            try:
                records.extend(_load_json_list(path))
            except Exception as e:
                sys.stderr.write(f"warn: skipping {path}: {e}\n")
    return records


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    series = pd.Series(values).dropna()
    if series.empty:
        return float("nan"), float("nan"), float("nan")
    return float(series.median()), float(series.quantile(0.25)), float(series.quantile(0.75))


def aggregate_training(records: list[dict]) -> TrainAggregate:
    energies: list[float] = []
    co2s: list[float] = []
    for r in records:
        e = r.get("energy_wh")
        if e is not None:
            energies.append(float(e))
        c = r.get("co2_g_total")
        if c is None:
            c = r.get("co2_g_operational")
        if c is not None:
            co2s.append(float(c))
    e_med, e_25, e_75 = _percentiles(energies)
    c_med, c_25, c_75 = _percentiles(co2s)
    hw = None
    for r in records:
        if r.get("hardware_id"):
            hw = r["hardware_id"]
            break
    return TrainAggregate(
        energy_wh_median=e_med,
        energy_wh_p25=e_25,
        energy_wh_p75=e_75,
        co2_g_median=c_med,
        co2_g_p25=c_25,
        co2_g_p75=c_75,
        n_seeds=len(records),
        hardware_id=hw,
        raw=records,
    )


def _safe_div(num: float, den: float) -> float:
    if math.isnan(num) or math.isnan(den):
        return float("inf")
    if den <= EPS:
        return float("inf")
    return num / den


def _size_from_filename(path: str) -> Optional[int]:
    name = os.path.basename(path)
    base = name.split(".")[0]
    for prefix in ("test_", "val_"):
        if base.startswith(prefix):
            try:
                return int(base[len(prefix):])
            except ValueError:
                return None
    return None


def build_aet_table(
    train_agg: TrainAggregate,
    inference_records: list[dict],
    baseline_records: list[dict],
    deltas: list[float],
    use_embodied_for_co2: bool = True,
) -> pd.DataFrame:
    """Cross every inference record with matching baseline records per (size).

    The join key is `size`: inference records do not encode size directly, so
    we expect the variant or test_file name to imply it. When ambiguous, all
    matching baselines are tried.
    """
    bdf = pd.DataFrame(baseline_records)
    if not bdf.empty:
        if "size" not in bdf.columns and "file" in bdf.columns:
            bdf["size"] = [
                _size_from_filename(str(f)) for f in bdf["file"].fillna("").tolist()
            ]
        if "file" in bdf.columns:
            bdf["size_key"] = [
                _size_from_filename(str(f)) for f in bdf["file"].fillna("").tolist()
            ]
        else:
            bdf["size_key"] = bdf.get("size")
        if "baseline_gap" in bdf.columns:
            bdf["baseline_gap"] = bdf["baseline_gap"].fillna(0.0)
        else:
            bdf["baseline_gap"] = 0.0
        bdf["E_meta_per_inst"] = bdf.apply(
            lambda r: _safe_div(
                float(r.get("energy_wh") or 0.0),
                float(r.get("num_problems") or 0.0),
            ),
            axis=1,
        )
        co2_col_meta = "co2_g_total" if use_embodied_for_co2 else "co2_g_operational"
        bdf["C_meta_per_inst"] = bdf.apply(
            lambda r: _safe_div(
                float(r.get(co2_col_meta)
                      or r.get("co2_g_total")
                      or r.get("co2_g_operational")
                      or 0.0),
                float(r.get("num_problems") or 0.0),
            ),
            axis=1,
        )

    idf = pd.DataFrame(inference_records)
    if idf.empty:
        return pd.DataFrame()
    if "test_file" in idf.columns:
        idf["size_key"] = [
            _size_from_filename(str(f)) for f in idf["test_file"].fillna("").tolist()
        ]
    else:
        idf["size_key"] = None
    if "size" not in idf.columns:
        idf["size"] = idf["size_key"]
    co2_col_nn = "co2_g_total" if use_embodied_for_co2 else "co2_g_operational"

    def _c_nn(r) -> float:
        n = r.get("n_instances")
        if not n:
            return float("nan")
        c = (r.get(co2_col_nn)
             or r.get("co2_g_total")
             or r.get("co2_g_operational")
             or 0.0)
        return float(c) / float(n)

    idf["C_NN_per_inst"] = idf.apply(_c_nn, axis=1)
    idf["E_NN_per_inst"] = idf.get("energy_wh_per_item")

    rows: list[dict] = []
    for _, inf in idf.iterrows():
        if bdf.empty:
            continue
        size_val = inf.get("size_key")
        if size_val is not None and "size_key" in bdf.columns:
            b_subset = bdf[bdf["size_key"] == size_val]
            if b_subset.empty:
                b_subset = bdf
        else:
            b_subset = bdf
        for _, base in b_subset.iterrows():
            for delta in deltas:
                baseline_gap = float(base.get("baseline_gap") or 0.0)
                nn_gap = float(inf.get("gap_to_bks") or 0.0)
                feasible = nn_gap <= baseline_gap + delta

                e_meta = float(base.get("E_meta_per_inst") or 0.0)
                e_nn = float(inf.get("E_NN_per_inst") or 0.0)
                c_meta = float(base.get("C_meta_per_inst") or 0.0)
                c_nn = float(inf.get("C_NN_per_inst") or 0.0)

                aet_E = _safe_div(train_agg.energy_wh_median, e_meta - e_nn)
                aet_C = _safe_div(train_agg.co2_g_median, c_meta - c_nn)
                aet_E_p25 = _safe_div(train_agg.energy_wh_p25, e_meta - e_nn)
                aet_E_p75 = _safe_div(train_agg.energy_wh_p75, e_meta - e_nn)

                if not feasible:
                    aet_E = float("inf")
                    aet_C = float("inf")
                    aet_E_p25 = float("inf")
                    aet_E_p75 = float("inf")

                rows.append(
                    {
                        "variant": inf.get("variant"),
                        "size": size_val,
                        "batch_size": inf.get("batch_size"),
                        "threads_mode": base.get("thread_mode"),
                        "num_procs": base.get("num_procs"),
                        "baseline_max_runtime_s": float(
                            base.get("max_runtime_s") or 0.0
                        ),
                        "delta_pct": delta,
                        "nn_gap_pct": nn_gap,
                        "baseline_gap_pct": baseline_gap,
                        "feasible": bool(feasible),
                        "hardware_id": inf.get("hardware_id"),
                        "seed": inf.get("seed"),
                        "throughput_items_per_s": inf.get("throughput_items_per_s"),
                        "E_train_wh_median": train_agg.energy_wh_median,
                        "E_NN_wh_per_inst": e_nn,
                        "E_meta_wh_per_inst": e_meta,
                        "C_train_g_median": train_agg.co2_g_median,
                        "C_NN_g_per_inst": c_nn,
                        "C_meta_g_per_inst": c_meta,
                        "aet_E": aet_E,
                        "aet_E_p25": aet_E_p25,
                        "aet_E_p75": aet_E_p75,
                        "aet_C": aet_C,
                    }
                )
    return pd.DataFrame(rows)


def plot_aet_by_delta(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty:
        ax.set_title("AET vs delta — no data")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    for key, sub in df.groupby(["batch_size", "threads_mode"]):
        batch, threads = key if isinstance(key, tuple) else (key, "?")
        sub = sub.sort_values("delta_pct")
        finite = sub[sub["feasible"] & sub["aet_E"].apply(lambda x: math.isfinite(x))]
        if finite.empty:
            continue
        ax.plot(finite["delta_pct"], finite["aet_E"], marker="o",
                label=f"batch={batch}, {threads}")
    ax.set_xlabel("δ (tolerated gap, %)")
    ax.set_ylabel("AET_E (instances)")
    ax.set_yscale("log")
    ax.set_title("AET vs δ")
    ax.grid(True, which="both", alpha=0.3)
    if ax.has_data():
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aet_by_batch(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty:
        ax.set_title("AET vs batch — no data")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    for key, sub in df.groupby(["delta_pct", "threads_mode"]):
        delta, threads = key if isinstance(key, tuple) else (key, "?")
        sub = sub.sort_values("batch_size")
        finite = sub[sub["feasible"] & sub["aet_E"].apply(lambda x: math.isfinite(x))]
        if finite.empty:
            continue
        ax.plot(finite["batch_size"], finite["aet_E"], marker="o",
                label=f"δ={delta}%, {threads}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("AET_E (instances)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("AET vs batch size")
    ax.grid(True, which="both", alpha=0.3)
    if ax.has_data():
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aet_by_size(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty or "size" not in df.columns:
        ax.set_title("AET vs size — no data")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    sub = df[df["feasible"]]
    if sub.empty:
        ax.set_title("AET vs size — no feasible combination")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    grouped = sub.groupby(["size", "threads_mode"]).agg(
        aet_med=("aet_E", "median"),
        aet_p25=("aet_E", lambda s: s.quantile(0.25) if len(s) else float("nan")),
        aet_p75=("aet_E", lambda s: s.quantile(0.75) if len(s) else float("nan")),
    ).reset_index()
    for threads, g in grouped.groupby("threads_mode"):
        g = g.sort_values("size")
        ax.plot(g["size"], g["aet_med"], marker="o", label=f"{threads}")
        ax.fill_between(g["size"], g["aet_p25"], g["aet_p75"], alpha=0.2)
    ax.set_xlabel("Problem size")
    ax.set_ylabel("AET_E (instances)")
    ax.set_yscale("log")
    ax.set_title("AET vs size (median + IQR multi-seed)")
    ax.grid(True, which="both", alpha=0.3)
    if ax.has_data():
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threads_bars(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty:
        ax.set_title("AET mono vs multi — no data")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    sub = df[df["feasible"]].copy()
    if sub.empty:
        ax.set_title("AET mono vs multi — no feasible combination")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    pivot = sub.groupby(["size", "threads_mode"])["aet_E"].median().unstack()
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Problem size")
    ax.set_ylabel("AET_E (instances) — median")
    ax.set_yscale("log")
    ax.set_title("AET by baseline thread budget")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_asymptotic_overview(df: pd.DataFrame, train_agg: TrainAggregate,
                             out_path: str) -> None:
    """Plot cumulative training+inference energy vs N for NN and the baseline."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty:
        ax.set_title("Asymptotic regime — no data")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    N = [10 ** k for k in range(0, 10)]
    feas = df[df["feasible"]]
    if feas.empty:
        ax.set_title("Asymptotic regime — no feasible combination")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    e_nn = float(pd.Series(feas["E_NN_wh_per_inst"]).median())
    e_meta = float(pd.Series(feas["E_meta_wh_per_inst"]).median())
    e_train = train_agg.energy_wh_median

    nn_curve = [(e_train or 0.0) + (e_nn or 0.0) * n for n in N]
    meta_curve = [(e_meta or 0.0) * n for n in N]
    ax.plot(N, nn_curve, marker="o", label=f"NN (E_train={e_train:.1f} Wh + N x {e_nn:.2e})")
    ax.plot(N, meta_curve, marker="s", label=f"Metaheuristic (N x {e_meta:.2e})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (deployed instances)")
    ax.set_ylabel("Cumulative energy (Wh)")
    ax.set_title("Asymptotic regime: NN vs metaheuristic")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="AET evaluation script")
    p.add_argument("--train-log", action="append", default=[],
                   help="energy_train.json file(s); accepts globs.")
    p.add_argument("--inference-log", action="append", default=[],
                   help="energy_inference_*.json file(s); accepts globs.")
    p.add_argument("--baseline-log", action="append", default=[],
                   help="energy_baseline_*.json file(s); accepts globs.")
    p.add_argument("--deltas", default="0,1,2,5,10",
                   help="Comma-separated delta percentages.")
    p.add_argument("--output-dir", default="docs/aet/figures",
                   help="Directory for CSV and plots.")
    p.add_argument("--no-embodied", action="store_true",
                   help="Use operational-only CO2 (exclude embodied carbon).")
    args = p.parse_args()

    train_records = _load_many(args.train_log)
    inference_records = _load_many(args.inference_log)
    baseline_records = _load_many(args.baseline_log)

    if not train_records:
        sys.stderr.write("error: no training records loaded\n")
        return 2
    if not inference_records:
        sys.stderr.write("error: no inference records loaded\n")
        return 2
    if not baseline_records:
        sys.stderr.write("error: no baseline records loaded\n")
        return 2

    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    train_agg = aggregate_training(train_records)
    df = build_aet_table(
        train_agg,
        inference_records,
        baseline_records,
        deltas=deltas,
        use_embodied_for_co2=not args.no_embodied,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "aet_results.csv")
    df.to_csv(csv_path, index=False)
    n_feasible = (
        int(pd.Series(df["feasible"]).sum())
        if not df.empty and "feasible" in df.columns
        else 0
    )
    print(f"wrote {csv_path}: {len(df)} rows, {n_feasible} feasible")

    plot_aet_by_delta(df, os.path.join(args.output_dir, "aet_by_delta.png"))
    plot_aet_by_batch(df, os.path.join(args.output_dir, "aet_by_batch.png"))
    plot_aet_by_size(df, os.path.join(args.output_dir, "aet_by_size.png"))
    plot_threads_bars(df, os.path.join(args.output_dir, "aet_threads_bars.png"))
    plot_asymptotic_overview(df, train_agg,
                             os.path.join(args.output_dir, "asymptotic_overview.png"))

    summary = {
        "train": {
            "energy_wh_median": train_agg.energy_wh_median,
            "energy_wh_p25": train_agg.energy_wh_p25,
            "energy_wh_p75": train_agg.energy_wh_p75,
            "co2_g_median": train_agg.co2_g_median,
            "n_seeds": train_agg.n_seeds,
            "hardware_id": train_agg.hardware_id,
        },
        "deltas": deltas,
        "n_inference_records": len(inference_records),
        "n_baseline_records": len(baseline_records),
        "n_aet_rows": int(len(df)),
        "n_feasible": int(pd.Series(df["feasible"]).sum()) if not df.empty else 0,
    }
    summary_path = os.path.join(args.output_dir, "aet_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
