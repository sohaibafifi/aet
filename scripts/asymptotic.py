"""Asymptotic regime analysis: plot the structural argument.

For a neural solver and a metaheuristic baseline:

    E_total_NN(N)   = E_train + N * E_NN_per_inst
    E_total_meta(N) = N * E_meta_per_inst

When `E_NN_per_inst < E_meta_per_inst`, the ratio
`E_total_NN(N) / E_total_meta(N) -> E_NN_per_inst / E_meta_per_inst < 1`
as `N -> infinity`. The training cost amortizes; the marginal advantage is
structural.

This script generates the canonical figure for the rebuttal blog post.

Two modes:
  (a) Parametric (default): plug in E_train, E_NN_per_inst, E_meta_per_inst
      directly via CLI flags. Useful for back-of-envelope / honest assumption.
  (b) Data-driven: read the AET CSV produced by `aet_eval.py` and pull the
      medians from there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from typing import Optional


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"matplotlib required: {e}\n")
    raise


def crossover_N(e_train: float, e_nn: float, e_meta: float) -> Optional[float]:
    """Return the break-even N (a.k.a. AET) for a given parameter triple."""
    denom = e_meta - e_nn
    if denom <= 0:
        return None
    return e_train / denom


def asymptotic_ratio(e_nn: float, e_meta: float) -> Optional[float]:
    if e_meta <= 0:
        return None
    return e_nn / e_meta


def plot_energy_curves(
    e_train: float,
    e_nn: float,
    e_meta: float,
    e_train_p25: Optional[float],
    e_train_p75: Optional[float],
    out_path: str,
    title: str = "Asymptotic regime: neural solver vs metaheuristic",
    xlabel: str = "N (deployed instances)",
    ylabel: str = "Cumulative energy (Wh)",
) -> dict:
    N = [10 ** k for k in range(0, 10)]
    nn_curve = [e_train + e_nn * n for n in N]
    meta_curve = [e_meta * n for n in N]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        N, nn_curve, marker="o", linewidth=2, color="#1f77b4",
        label=f"NN  (E_train={e_train:.1f} Wh, E_NN={e_nn:.2e} Wh/inst)",
    )
    if e_train_p25 is not None and e_train_p75 is not None:
        nn_lo = [e_train_p25 + e_nn * n for n in N]
        nn_hi = [e_train_p75 + e_nn * n for n in N]
        ax.fill_between(N, nn_lo, nn_hi, alpha=0.18, color="#1f77b4",
                        label="NN (IQR over training seeds)")
    ax.plot(
        N, meta_curve, marker="s", linewidth=2, color="#d62728",
        label=f"Metaheuristic  (E_meta={e_meta:.2e} Wh/inst)",
    )

    crossover = crossover_N(e_train, e_nn, e_meta)
    if crossover is not None and crossover > 0:
        ax.axvline(crossover, linestyle="--", color="black", alpha=0.5)
        ax.annotate(
            f"AET ≈ {crossover:,.0f}",
            xy=(crossover, e_train),
            xytext=(crossover * 1.5, e_train * 1.5),
            arrowprops=dict(arrowstyle="->", color="black", alpha=0.5),
            fontsize=10,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "crossover_N": crossover,
        "asymptotic_ratio": asymptotic_ratio(e_nn, e_meta),
        "e_train_wh": e_train,
        "e_nn_wh_per_inst": e_nn,
        "e_meta_wh_per_inst": e_meta,
        "output_path": out_path,
    }


def plot_ratio_curve(
    e_train: float,
    e_nn: float,
    e_meta: float,
    out_path: str,
) -> None:
    """Plot E_total_NN(N) / E_total_meta(N), showing it tends to e_nn/e_meta."""
    N = [10 ** k for k in range(0, 10)]
    ratios = []
    for n in N:
        nn = e_train + e_nn * n
        meta = e_meta * n
        if meta <= 0:
            ratios.append(float("nan"))
        else:
            ratios.append(nn / meta)

    asymp = asymptotic_ratio(e_nn, e_meta)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(N, ratios, marker="o", linewidth=2, color="#2ca02c",
            label="E_total_NN / E_total_meta")
    if asymp is not None:
        ax.axhline(asymp, linestyle="--", color="black", alpha=0.6,
                   label=f"asymptote = {asymp:.2e}")
    ax.axhline(1.0, linestyle=":", color="gray", alpha=0.7,
               label="parity (NN = meta)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (deployed instances)")
    ax.set_ylabel("Cumulative energy ratio")
    ax.set_title("Asymptotic convergence of NN / metaheuristic ratio")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def from_csv(csv_path: str) -> Optional[dict]:
    try:
        import pandas as pd
    except Exception as e:
        sys.stderr.write(f"pandas required for --from-csv: {e}\n")
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    feas = df[df["feasible"]] if "feasible" in df.columns else df
    if feas.empty:
        sys.stderr.write("warn: no feasible rows in CSV\n")
        feas = df
    return {
        "e_train": float(pd.Series(feas["E_train_wh_median"]).median()),
        "e_train_p25": float(pd.Series(feas["E_train_wh_median"]).quantile(0.25))
        if "E_train_wh_median" in feas.columns else None,
        "e_train_p75": float(pd.Series(feas["E_train_wh_median"]).quantile(0.75))
        if "E_train_wh_median" in feas.columns else None,
        "e_nn": float(pd.Series(feas["E_NN_wh_per_inst"]).median()),
        "e_meta": float(pd.Series(feas["E_meta_wh_per_inst"]).median()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="AET asymptotic-regime plotter")
    p.add_argument("--e-train-wh", type=float, default=None,
                   help="Median training energy (Wh).")
    p.add_argument("--e-nn-wh", type=float, default=None,
                   help="Per-instance NN inference energy (Wh).")
    p.add_argument("--e-meta-wh", type=float, default=None,
                   help="Per-instance metaheuristic energy (Wh).")
    p.add_argument("--e-train-p25", type=float, default=None)
    p.add_argument("--e-train-p75", type=float, default=None)
    p.add_argument("--from-csv", default=None,
                   help="aet_results.csv from aet_eval.py (overrides --e-*).")
    p.add_argument("--output-dir", default="docs/aet/figures")
    p.add_argument("--label", default="asymptotic",
                   help="Filename prefix for outputs.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.from_csv:
        params = from_csv(args.from_csv)
        if params is None:
            sys.stderr.write("error: could not derive params from CSV\n")
            return 2
        e_train = params["e_train"]
        e_train_p25 = params.get("e_train_p25")
        e_train_p75 = params.get("e_train_p75")
        e_nn = params["e_nn"]
        e_meta = params["e_meta"]
    else:
        if args.e_train_wh is None or args.e_nn_wh is None or args.e_meta_wh is None:
            sys.stderr.write(
                "error: provide --e-train-wh, --e-nn-wh and --e-meta-wh "
                "or --from-csv\n"
            )
            return 2
        e_train = args.e_train_wh
        e_train_p25 = args.e_train_p25
        e_train_p75 = args.e_train_p75
        e_nn = args.e_nn_wh
        e_meta = args.e_meta_wh

    curves_path = os.path.join(args.output_dir, f"{args.label}_curves.png")
    ratio_path = os.path.join(args.output_dir, f"{args.label}_ratio.png")
    summary = plot_energy_curves(
        e_train=e_train,
        e_nn=e_nn,
        e_meta=e_meta,
        e_train_p25=e_train_p25,
        e_train_p75=e_train_p75,
        out_path=curves_path,
    )
    plot_ratio_curve(e_train=e_train, e_nn=e_nn, e_meta=e_meta, out_path=ratio_path)
    summary["ratio_path"] = ratio_path

    summary_path = os.path.join(args.output_dir, f"{args.label}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
