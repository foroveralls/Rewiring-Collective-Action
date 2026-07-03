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
    python convergence_diagnostic.py            # simulate + plot (ALL points/conditions)
    python convergence_diagnostic.py --mode plot --data ../Output/<file>.pkl.gz

Partial reruns (--points/--conditions restrict what is simulated; --merge keeps
every record from an existing pkl that is not re-simulated, so outputs always
cover all 60 combinations). After the 2026-07-03 horizon changes the required
revalidation is (see claude_stuff/convergence_diagnostic_criteria_2026-07-03.md):

    # 1. stress point at the stubbornness-scaled horizons (expensive: ~26.4M
    #    steps x 20 runs across 30 conditions; heaviest condition ~1.3M steps)
    python convergence_diagnostic.py --points stress \
        --merge ../Output/convergence_diagnostic_2026-07-02.pkl.gz

    # 2. the six main-point conditions whose factors were raised, merged with
    #    step 1's dated output
    python convergence_diagnostic.py --points main \
        --conditions biased/diff/DPAH,bridge/diff/DPAH,bridge/diff/FB,random/None/cl,None/None/FB,wtf/None/Twitter \
        --merge ../Output/convergence_diagnostic_<step1-date>.pkl.gz

The 24 unchanged main-point conditions are already validated by the 2026-07-02
data and are carried over by --merge. Horizon factors do NOT read from these
outputs; if failures remain, update get_adaptive_timesteps in
general_param_sweep.py manually and validate again.
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
from scipy.optimize import curve_fit
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

# Plateau criterion (state lives on [-1, 1]). Both tests act on the region
# BEYOND the adaptive cutoff (the shaded area in the figure), and each counts
# as passed when the quantity is either practically negligible (below an
# equivalence margin) or statistically indistinguishable from zero at ~Z_CRIT
# standard errors. Drift uses a Geweke-style two-window comparison of the
# ensemble mean scaled by its Monte Carlo standard error (Geweke 1992); the
# post-cutoff slope gets a t-style test from OLS on batch means to blunt
# autocorrelation (Law 2015, ch. 9). Window-mean comparison follows
# Grazzini (2012), JASSS 15(2)7.
DRIFT_EPS = 0.05            # equivalence margin for drift beyond the cutoff: 2.5% of the
                            # state range, small vs. the between-condition differences
                            # (~0.1-0.5) analysed in the paper
SLOPE_EPS = 0.01            # |post-cutoff slope| per 10k steps that is negligible outright
Z_CRIT = 2.0                # SE multiple below which drift/slope are consistent with zero
N_BATCHES = 10              # batch means for the slope regression


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


def diag_horizon_for(algo, topo, mode, stubbornness=None):
    """Horizon this diagnostic runs a condition to (>= its adaptive horizon)."""
    adaptive = get_adaptive_timesteps(algo, topo, mode, stubbornness=stubbornness)
    if DIAG_HORIZON is not None:
        # never below the margin, so the validation region beyond the cutoff
        # always exists even when stubbornness scaling pushes the horizon up
        return max(int(DIAG_HORIZON), int(HORIZON_MARGIN * adaptive))
    return int(HORIZON_MARGIN * adaptive)


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

    horizon = diag_horizon_for(algo, topo, mode, point["stubbornness"])
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
        "adaptive_horizon": get_adaptive_timesteps(
            algo, topo, mode, stubbornness=point["stubbornness"]),
        "diag_horizon": horizon,
        "n_runs": N_RUNS,
    }


def plateau_metrics(rec):
    """Quantify convergence at/after the adaptive cutoff.

    Drift and slope are both measured on the post-cutoff region and compared
    against the trajectory's own noise scale rather than absolute thresholds
    alone: states are window means (final 10% of the adaptive window vs. the
    last window of the diagnostic run), drift is scaled by the Monte Carlo
    standard error of the ensemble mean, and the post-cutoff slope gets a
    t-style test from a batch-means regression.
    """
    mean, std = rec["mean"], rec["std"]
    H, L = rec["adaptive_horizon"], len(mean)
    n_runs = rec.get("n_runs", N_RUNS)
    h_idx = min(H, L - 1)
    w = max(2, h_idx // 10)                       # final 10% of the adaptive window
    lo = max(0, h_idx - w)
    cut_win = slice(lo, h_idx + 1)
    end_win = slice(max(0, L - w), L)

    state_at_cut = float(mean[cut_win].mean())
    state_at_end = float(mean[end_win].mean())
    drift_beyond = abs(state_at_end - state_at_cut)
    sem = float(np.sqrt((std[cut_win].mean() ** 2 + std[end_win].mean() ** 2) / 2)
                / np.sqrt(n_runs))
    drift_z = drift_beyond / sem if sem > 0 else np.inf

    # slope from batch means over the whole post-cutoff region (should be flat)
    tail = mean[h_idx:]
    xs = np.arange(h_idx, L, dtype=np.float64)
    nb = min(N_BATCHES, max(3, len(tail) // 2))
    edges = np.linspace(0, len(tail), nb + 1).astype(int)
    bx = np.array([xs[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    by = np.array([tail[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    slope_ps, intercept = np.polyfit(bx, by, 1)
    resid = by - (slope_ps * bx + intercept)
    slope_se_ps = np.sqrt((resid ** 2).sum() / (nb - 2)
                          / ((bx - bx.mean()) ** 2).sum())
    slope = float(slope_ps) * 1e4                 # per 10k steps
    slope_se = float(slope_se_ps) * 1e4
    tail_std = float(tail.std())

    drift_ok = (drift_beyond < DRIFT_EPS) or (drift_z < Z_CRIT)
    slope_ok = (abs(slope) < SLOPE_EPS) or (abs(slope) < Z_CRIT * slope_se)
    return {
        "state_at_cutoff": state_at_cut,
        "state_at_end": state_at_end,
        "drift_beyond_cutoff": drift_beyond,
        "drift_sem": sem,
        "drift_z": drift_z,
        "terminal_slope_per_10k": slope,
        "slope_se_per_10k": slope_se,
        "tail_std": tail_std,
        "converged": bool(drift_ok and slope_ok),
    }


# Asymptote fit: bounds on A reflect the physical state range; tau bounds and
# the R^2 floor flag fits where the trajectory has too little curvature to
# identify the asymptote (extrapolation is then meaningless, only a lower
# bound on the required horizon can be claimed).
FIT_TAU_MAX = 5e6
FIT_R2_MIN = 0.9


def asymptote_metrics(rec):
    """Fit x(t) = A - B*exp(-(t-t0)/tau) to the trajectory tail.

    Complements the plateau criterion: the fit yields the asymptote A, the
    relaxation timescale tau, and the horizon t_required at which the
    trajectory comes within DRIFT_EPS of A. asym_identified is False when the
    tail is too close to linear to pin the asymptote down (A or tau at their
    bounds, or poor fit), in which case t_required is a lower bound at best.
    """
    mean, H, L = rec["mean"], rec["adaptive_horizon"], len(rec["mean"])
    t0 = L // 4                                   # skip the initial transient
    t = np.arange(L, dtype=np.float64)[t0:]
    y = mean[t0:]

    def model(t, A, B, tau):
        return A - B * np.exp(-(t - t0) / tau)

    A0 = y[-max(1, len(y) // 10):].mean()
    try:
        p, _ = curve_fit(model, t, y, p0=[A0, A0 - y[0], len(y) / 3],
                         bounds=([-1.0, -3, 1e2], [1.0, 3, FIT_TAU_MAX]),
                         maxfev=20000)
    except RuntimeError:
        return {"asym_A": np.nan, "asym_tau": np.nan, "asym_r2": np.nan,
                "t_required": np.nan, "asym_identified": False,
                "horizon_sufficient_fit": False}
    A, B, tau = p
    yhat = model(t, *p)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yhat) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    t_req = t0 + tau * np.log(max(abs(B) / DRIFT_EPS, 1.0))
    identified = (r2 >= FIT_R2_MIN) and (abs(A) <= 0.995) and (tau <= 0.5 * FIT_TAU_MAX)
    return {
        "asym_A": float(A),
        "asym_tau": float(tau),
        "asym_r2": float(r2),
        "t_required": float(t_req),
        "asym_identified": bool(identified),
        "horizon_sufficient_fit": bool(identified and t_req <= H),
    }


def simulate_all(point_labels=None, condition_filter=None):
    """Simulate the diagnostic; optionally restrict to a subset.

    point_labels: iterable of PARAM_POINTS labels (e.g. {"stress"}), or None for all.
    condition_filter: set of "algo/mode/topo" strings, or None for all.
    """
    conditions = build_sweep_conditions()
    if condition_filter:
        conditions = [c for c in conditions
                      if f"{c[0]}/{c[1]}/{c[2]}" in condition_filter]
        missing = condition_filter - {f"{c[0]}/{c[1]}/{c[2]}" for c in conditions}
        assert not missing, f"Unknown conditions: {missing}"
    points = [p for p in PARAM_POINTS
              if point_labels is None or p["label"] in point_labels]
    assert points and conditions, "Nothing to simulate after filtering"

    base_args = models_checks.getargs()
    total = len(conditions) * len(points)
    print(f"=== CONVERGENCE DIAGNOSTIC ===")
    print(f"Conditions: {len(conditions)} x param points: {len(points)} "
          f"x runs: {N_RUNS}")
    print(f"Diagnostic horizon: {'uniform ' + str(DIAG_HORIZON) if DIAG_HORIZON else f'{HORIZON_MARGIN}x adaptive'}")
    print("=" * 40)

    data, done, start = {}, 0, time.time()
    for point in points:
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
    return report_table(data, tag)


def report_table(data, tag):
    rows = []
    for (label, algo, mode, topo), rec in data.items():
        m = plateau_metrics(rec)
        m.update(asymptote_metrics(rec))
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
                "drift_beyond_cutoff", "drift_z",
                "terminal_slope_per_10k", "slope_se_per_10k"]
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
            ax.annotate(f"drift={met['drift_beyond_cutoff']:.3f} (z={met['drift_z']:.1f})\n"
                        f"slope={met['terminal_slope_per_10k']:+.3f}"
                        f"$\\pm${met['slope_se_per_10k']:.3f}",
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
    ap.add_argument("--points", help="comma-separated param point labels to "
                    "simulate (e.g. 'stress'); default: all")
    ap.add_argument("--conditions", help="comma-separated algo/mode/topo to "
                    "simulate (e.g. 'biased/diff/DPAH,wtf/None/Twitter'); "
                    "default: all")
    ap.add_argument("--merge", help="existing .pkl.gz whose records are kept "
                    "for every condition NOT simulated in this run (combined "
                    "output saved under today's tag)")
    args = ap.parse_args()
    tag = str(date.today())

    if args.mode == "plot":
        assert args.data, "--mode plot requires --data <file.pkl.gz>"
        with gzip.open(args.data, "rb") as f:
            data = pickle.load(f)
        tag = os.path.basename(args.data).replace("convergence_diagnostic_", "").replace(".pkl.gz", "")
        report_table(data, tag)
        plot_diagnostic(data, tag)
        return

    points = set(args.points.split(",")) if args.points else None
    conds = set(args.conditions.split(",")) if args.conditions else None
    data = simulate_all(points, conds)
    if args.merge:
        with gzip.open(args.merge, "rb") as f:
            old = pickle.load(f)
        kept = {k: v for k, v in old.items() if k not in data}
        print(f"Merged {len(kept)} existing records from {args.merge}")
        data = {**kept, **data}
    save_and_report(data, tag)
    if args.mode == "all":
        plot_diagnostic(data, tag)


if __name__ == "__main__":
    main()
