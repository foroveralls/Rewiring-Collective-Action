"""
Convergence diagnostic for the sensitivity-sweep horizons.

The heatmap sweep (general_param_sweep.py) runs each (topology, algorithm, mode)
condition to a per-condition adaptive horizon (get_adaptive_timesteps). Those
factors were calibrated by eye from trajectory plots. This script produces the
evidence that each horizon is actually sufficient: it runs every sweep condition
PAST its adaptive cutoff (to a uniform diagnostic horizon) and shows the mean
opinion has plateaued by the cutoff, both visually and via a quantitative
criterion.

Deliverables (written to ../Output and ../Figs):
  - convergence_diagnostic_<date>.pkl.gz : full averaged trajectories + metadata
  - convergence_diagnostic_<date>.csv    : per-condition plateau table (SI)
  - convergence_diagnostic_<date>.png    : small-multiples plateau figure (SI)

Run from the Analysis/ directory (paths are relative, matching run_phased.py):
    python convergence_diagnostic.py            # simulate + plot
    python convergence_diagnostic.py --mode plot --data ../Output/<file>.pkl.gz
"""

import os
import gc
import time
import glob
import gzip
import pickle
import argparse
from datetime import date
from itertools import repeat

import numpy as np
import pandas as pd
import multiprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models_checks
# Single source of truth for the horizons + pool helpers used by the real sweep.
from general_param_sweep import (
    get_adaptive_timesteps,
    get_optimal_process_count,
    group_scenarios_by_algorithm,
    init,
)

# ----------------------------------------------------------------------------
# Configuration (edit here; kept as constants rather than CLI flags on purpose)
# ----------------------------------------------------------------------------
N_RUNS = 20                 # runs averaged per condition (smooths the mean; sweep uses 30)
DIAG_HORIZON = 160000       # uniform horizon to run every condition to (matches the
                            # main trajectory figure). Set to None to instead run each
                            # condition to HORIZON_MARGIN * its adaptive horizon, which
                            # is much cheaper for node2vec (the per-step bottleneck).
HORIZON_MARGIN = 1.6        # only used when DIAG_HORIZON is None

# Parameter points to check. The sweep varies (stubbornness, polarisingNode_f);
# stubbornness sets the relaxation timescale, so high stubbornness is the
# slowest-converging corner of the grid.
PARAM_POINTS = [
    {"label": "main",   "stubbornness": 0.6, "polarisingNode_f": 0.10, "seed": 42},
    {"label": "stress", "stubbornness": 0.9, "polarisingNode_f": 0.10, "seed": 43},
]

# Plateau criterion (state lives on [-1, 1]).
DRIFT_EPS = 0.01            # max |mean(end) - mean(cutoff)| to count as converged
SLOPE_EPS = 0.01           # max |terminal slope| per 10k steps to count as flat


def build_sweep_conditions():
    """Exactly the 30 conditions from general_param_sweep.combined_list."""
    rewiring_list_h = ["diff", "same"]
    directed = ["DPAH", "Twitter"]
    undirected = ["cl", "FB"]

    c1 = [(s, r, t) for s in ["biased", "bridge"]
          for r in rewiring_list_h for t in directed + undirected]
    c_rand = [("random", "None", t) for t in directed + undirected]
    c2 = [("node2vec", "None", t) for t in directed + undirected]
    c3 = [("None", "None", t) for t in directed + undirected]
    c4 = [("wtf", "None", t) for t in directed]
    return c1 + c_rand + c2 + c3 + c4


def diag_horizon_for(algo, topo, mode):
    """Horizon this diagnostic runs a condition to (>= its adaptive horizon)."""
    if DIAG_HORIZON is not None:
        return int(DIAG_HORIZON)
    return int(HORIZON_MARGIN * get_adaptive_timesteps(algo, topo, mode))


def _states_worker(i, sim_args):
    """Run one simulation, return only the mean-opinion trajectory (light IPC)."""
    res = models_checks.simulate(i, sim_args)
    model = res[0] if isinstance(res, tuple) else res
    return np.asarray(model.states, dtype=np.float64)


def run_condition(algo, mode, topo, point, base_args):
    """Average N_RUNS trajectories for one condition at one param point."""
    if topo == "Twitter":
        top_file, nwsize = "twitter_graph_N_789.gpickle", 789
    elif topo == "FB":
        top_file, nwsize = "FB_graph_N_786.gpickle", 786
    else:
        top_file, nwsize = None, 800

    horizon = diag_horizon_for(algo, topo, mode)
    sim_args = {
        **base_args,
        "rewiringAlgorithm": algo, "rewiringMode": mode, "type": topo,
        "top_file": top_file, "nwsize": nwsize, "timesteps": horizon,
        "stubbornness": point["stubbornness"],
        "polarisingNode_f": point["polarisingNode_f"],
        "plot": False, "save_snapshots": False, "seed": point["seed"],
    }

    nproc = get_optimal_process_count()
    lock = multiprocessing.Lock()
    with multiprocessing.Pool(processes=nproc, initializer=init, initargs=(lock,)) as pool:
        trajs = pool.starmap(_states_worker,
                             zip(range(N_RUNS), repeat(sim_args)))
        pool.close()
        pool.join()
    gc.collect()

    L = min(len(t) for t in trajs)
    arr = np.vstack([t[:L] for t in trajs])
    return {
        "t": np.arange(L),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "adaptive_horizon": get_adaptive_timesteps(algo, topo, mode),
        "diag_horizon": horizon,
        "n_runs": N_RUNS,
    }


def plateau_metrics(rec):
    """Quantify convergence at/after the adaptive cutoff."""
    mean, H, L = rec["mean"], rec["adaptive_horizon"], len(rec["mean"])
    h_idx = min(H, L - 1)
    w = max(2, h_idx // 10)                       # final 10% of the adaptive window
    lo = max(0, h_idx - w)

    state_at_cut = float(mean[h_idx])
    state_at_end = float(mean[-1])
    drift_beyond = abs(state_at_end - state_at_cut)

    x = np.arange(lo, h_idx + 1)
    slope = float(np.polyfit(x, mean[lo:h_idx + 1], 1)[0]) * 1e4  # per 10k steps
    tail_std = float(mean[lo:h_idx + 1].std())

    converged = (drift_beyond < DRIFT_EPS) and (abs(slope) < SLOPE_EPS)
    return {
        "state_at_cutoff": state_at_cut,
        "state_at_end": state_at_end,
        "drift_beyond_cutoff": drift_beyond,
        "terminal_slope_per_10k": slope,
        "tail_std": tail_std,
        "converged": bool(converged),
    }


def simulate_all():
    conditions = build_sweep_conditions()
    base_args = models_checks.getargs()
    total = len(conditions) * len(PARAM_POINTS)
    print(f"=== CONVERGENCE DIAGNOSTIC ===")
    print(f"Conditions: {len(conditions)} x param points: {len(PARAM_POINTS)} "
          f"x runs: {N_RUNS}")
    print(f"Diagnostic horizon: {'uniform ' + str(DIAG_HORIZON) if DIAG_HORIZON else f'{HORIZON_MARGIN}x adaptive'}")
    print("=" * 40)

    data, done, start = {}, 0, time.time()
    for point in PARAM_POINTS:
        for algo, mode, topo in conditions:
            t0 = time.time()
            rec = run_condition(algo, mode, topo, point, base_args)
            data[(point["label"], algo, mode, topo)] = rec
            done += 1
            print(f"[{done}/{total}] {point['label']:>6} | {algo}/{mode}/{topo} "
                  f"-> {rec['diag_horizon']} steps in {(time.time()-t0)/60:.1f} min "
                  f"(cutoff @ {rec['adaptive_horizon']})")

    for f in glob.glob("*embeddings*"):
        os.remove(f)
    print(f"Total simulate time: {(time.time()-start)/3600:.2f} h")
    return data


def save_and_report(data, tag):
    pkl = f"../Output/convergence_diagnostic_{tag}.pkl.gz"
    with gzip.open(pkl, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved trajectories: {pkl}")

    rows = []
    for (label, algo, mode, topo), rec in data.items():
        m = plateau_metrics(rec)
        rows.append({"param_point": label, "algorithm": algo, "mode": mode,
                     "topology": topo, "adaptive_horizon": rec["adaptive_horizon"],
                     "diag_horizon": rec["diag_horizon"], **m})
    df = pd.DataFrame(rows)
    csv = f"../Output/convergence_diagnostic_{tag}.csv"
    df.to_csv(csv, index=False)
    n_fail = int((~df["converged"]).sum())
    print(f"Saved plateau table: {csv}")
    print(f"Conditions NOT converged by their cutoff: {n_fail}/{len(df)}")
    if n_fail:
        cols = ["param_point", "algorithm", "mode", "topology",
                "drift_beyond_cutoff", "terminal_slope_per_10k"]
        print(df.loc[~df["converged"], cols].to_string(index=False))
    return df


def plot_diagnostic(data, tag):
    # One figure per param point; conditions ordered as in the sweep.
    order = build_sweep_conditions()
    for label in sorted({k[0] for k in data}):
        recs = [((label, a, m, t), data[(label, a, m, t)])
                for (a, m, t) in order if (label, a, m, t) in data]
        n = len(recs)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")

        for idx, ((_, algo, mode, topo), rec) in enumerate(recs):
            ax = axes[idx // ncols][idx % ncols]
            ax.axis("on")
            t, mean, std = rec["t"], rec["mean"], rec["std"]
            H = rec["adaptive_horizon"]
            met = plateau_metrics(rec)
            ok = met["converged"]

            ax.fill_between(t, mean - std, mean + std, color="#4C72B0", alpha=0.18,
                            linewidth=0)
            ax.plot(t, mean, color="#4C72B0", lw=1.2)
            # region beyond the adaptive cutoff (should be flat)
            ax.axvspan(H, t[-1], color="0.85", alpha=0.5, zorder=0)
            ax.axvline(H, color=("#2ca02c" if ok else "#d62728"), ls="--", lw=1.2)

            name = f"{algo}" + (f"/{mode}" if mode != "None" else "") + f" · {topo}"
            ax.set_title(name, fontsize=8.5)
            ax.annotate(f"drift={met['drift_beyond_cutoff']:.3f}\n"
                        f"slope={met['terminal_slope_per_10k']:+.3f}",
                        xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=6.5,
                        color=("#2ca02c" if ok else "#d62728"))
            ax.set_xlim(0, t[-1])
            ax.tick_params(labelsize=6.5)
            ax.set_xlabel("timestep", fontsize=7)
            ax.set_ylabel("mean opinion", fontsize=7)

        fig.suptitle(f"Convergence diagnostic — param point: {label}\n"
                     f"dashed line = adaptive sweep horizon; shaded = beyond cutoff "
                     f"(green = converged, red = not)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = f"../Figs/convergence_diagnostic_{tag}_{label}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved figure: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "plot", "all"], default="all")
    ap.add_argument("--data", help="existing .pkl.gz to plot (for --mode plot)")
    args = ap.parse_args()
    tag = str(date.today())

    if args.mode == "plot":
        assert args.data, "--mode plot requires --data <file.pkl.gz>"
        with gzip.open(args.data, "rb") as f:
            data = pickle.load(f)
        tag = os.path.basename(args.data).replace("convergence_diagnostic_", "").replace(".pkl.gz", "")
        plot_diagnostic(data, tag)
        return

    data = simulate_all()
    save_and_report(data, tag)
    if args.mode == "all":
        plot_diagnostic(data, tag)


if __name__ == "__main__":
    main()
