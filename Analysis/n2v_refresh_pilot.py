# -*- coding: utf-8 -*-
"""
node2vec re-embedding pilot: rerun the default-parameter node2vec scenarios with
the embedding retrained on the live graph every R steps (n2v_refresh = R), against
the canonical static arm (one embedding, trained at t = 0).

Background, decision rule and expected cost:
claude_stuff/Review/n2v_refresh_pilot_2026-10-12.md

All arms run here, on the same code and machine, so the comparison is
self-contained; the static arm is also checked against the 90 canonical runs of
the main campaign. Needs the SNAP node2vec binary at Auxillary/node2vec.

Usage:
    cd Analysis
    python n2v_refresh_pilot.py --timing   # seconds per retrain, projected pilot cost
    python n2v_refresh_pilot.py --smoke    # quick end-to-end validation
    python n2v_refresh_pilot.py            # full pilot
"""

import sys
import time
import multiprocessing
from datetime import date
from itertools import repeat

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la

import models_checks
from general_param_sweep import get_optimal_process_count, init

# %% configuration
N_SIMS = 30
REFRESH = [0, 8000, 800]  # 0 = canonical static embedding; 800 steps = one sweep of N = 800
TOPOLOGIES = ["DPAH", "Twitter", "cl", "FB"]
TIMESTEPS = 160000  # main-campaign node2vec horizon (run_phased.py)
TRAJ_EVERY = 1000  # stride of the saved mean-opinion trajectory

TOP_FILES = {"Twitter": ("twitter_graph_N_789.gpickle", 789),
             "FB": ("FB_graph_N_786.gpickle", 786)}


# %% network measures (same conventions as extract_network_metrics_lowmem.py:
# Leiden, unweighted, seed 42, direction preserved; in-degree Gini)
def _modularity(g):
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in g.edges()],
                    directed=g.is_directed())
    return la.find_partition(ig_g, la.ModularityVertexPartition, seed=42).modularity


def _gini(values):
    x = np.sort(np.asarray(list(values), dtype=float))
    n, total = x.size, x.sum()
    if n == 0 or total <= 0:
        return np.nan
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def _in_degrees(g):
    return [d for _, d in (g.in_degree() if g.is_directed() else g.degree())]


def _run(i, sim_args):
    """Worker: one run, reduced to scalars and a strided trajectory so that the
    model (and its graph) never travels back through the pool."""
    t0 = time.time()
    model = models_checks.simulate(i, sim_args)
    wall = time.time() - t0
    g = model.graph
    return {
        "state": model.states[-1], "state_std": model.statesds[-1],
        "aligned": model.ratio[-1],
        "modularity": _modularity(g), "gini_in_degree": _gini(_in_degrees(g)),
        "n_edges": g.number_of_edges(), "retrains": model.retrain, "wall_s": wall,
        "traj": np.asarray(model.states[::TRAJ_EVERY], dtype=np.float32),
    }


def run_scenario(pool, refresh, topo, n_sims, base_args, timesteps):
    top_file, nwsize = TOP_FILES.get(topo, (None, 800))
    sim_args = {**base_args,
                "rewiringAlgorithm": "node2vec", "rewiringMode": "None", "type": topo,
                "top_file": top_file, "nwsize": nwsize, "timesteps": timesteps,
                "polarisingNode_f": 0.10, "n2v_refresh": refresh,
                "plot": False, "save_snapshots": False, "seed": 42}
    out = pool.starmap(_run, zip(range(n_sims), repeat(sim_args)))

    rows, traj = [], []
    for i, r in enumerate(out):
        expected = (timesteps - 1) // refresh if refresh else 0
        if r["retrains"] < 0.9 * expected:
            print(f"  WARNING run {i}: {r['retrains']} retrains, expected ~{expected}")
        series = r.pop("traj")
        rows.append({"refresh": refresh, "topology": topo, "sim": i,
                     "timesteps": timesteps, **r})
        traj.extend({"refresh": refresh, "topology": topo, "sim": i,
                     "t": k * TRAJ_EVERY, "state": float(v)} for k, v in enumerate(series))
    return rows, traj


def timing(base_args, n_procs):
    """One short run per topology with and without refresh; the difference over the
    number of retrains is the cost of one retrain, which sets the pilot budget."""
    steps, refresh = 4000, 400
    print(f"=== N2V REFRESH TIMING ({steps} steps, refresh every {refresh}) ===")
    total_s = 0.0
    with multiprocessing.Pool(processes=2, initializer=init,
                              initargs=(multiprocessing.Lock(),)) as pool:
        for topo in TOPOLOGIES:
            top_file, nwsize = TOP_FILES.get(topo, (None, 800))
            args = {**base_args, "rewiringAlgorithm": "node2vec", "rewiringMode": "None",
                    "type": topo, "top_file": top_file, "nwsize": nwsize,
                    "timesteps": steps, "polarisingNode_f": 0.10, "plot": False}
            static, fresh = pool.starmap(_run, [(0, {**args, "n2v_refresh": 0}),
                                                (0, {**args, "n2v_refresh": refresh})])
            per = (fresh["wall_s"] - static["wall_s"]) / max(fresh["retrains"], 1)
            base = (static["wall_s"] - per) * TIMESTEPS / steps + per  # one initial training
            print(f"  {topo:8s} static {static['wall_s']:.1f}s, refreshed {fresh['wall_s']:.1f}s, "
                  f"{fresh['retrains']} retrains -> {per:.2f} s/retrain")
            for R in REFRESH:
                run_s = base + (per * TIMESTEPS / R if R else 0)
                total_s += N_SIMS * run_s
                print(f"           R={R:5d}: ~{run_s/60:.0f} min/run at {TIMESTEPS} steps")
    print(f"Projected full pilot: {total_s/3600:.0f} CPU-h, "
          f"~{total_s/3600/n_procs:.1f} h wall on {n_procs} processes "
          f"(full retrains use 2 threads each)")


if __name__ == "__main__":
    base_args = models_checks.getargs()
    n_procs = get_optimal_process_count()

    if "--timing" in sys.argv:
        timing(base_args, n_procs)
        sys.exit(0)

    smoke = "--smoke" in sys.argv
    n_sims = 2 if smoke else N_SIMS
    timesteps = 2000 if smoke else TIMESTEPS
    refresh_arms = [0, 500] if smoke else REFRESH
    topologies = ["DPAH", "cl"] if smoke else TOPOLOGIES
    if smoke:
        n_procs = 2

    total = len(refresh_arms) * len(topologies) * n_sims
    print(f"=== N2V REFRESH PILOT {'(smoke) ' if smoke else ''}===")
    print(f"Date: {date.today()} | arms {refresh_arms} x {topologies} x {n_sims} sims "
          f"= {total} runs at {timesteps} steps | processes: {n_procs}")

    start = time.time()
    rows, traj = [], []
    with multiprocessing.Pool(processes=n_procs, initializer=init,
                              initargs=(multiprocessing.Lock(),)) as pool:
        for topo in topologies:
            for refresh in refresh_arms:
                t0 = time.time()
                r, tr = run_scenario(pool, refresh, topo, n_sims, base_args, timesteps)
                rows.extend(r)
                traj.extend(tr)
                print(f"  {topo} R={refresh}: {(time.time()-t0)/60:.1f} min")

    suffix = "_smoke" if smoke else ""
    df = pd.DataFrame(rows)
    fname = f"../Output/n2v_refresh_pilot_{date.today()}{suffix}.csv"
    df.to_csv(fname, index=False)
    pd.DataFrame(traj).to_csv(fname.replace(".csv", "_traj.csv"), index=False)
    print(f"\nDone in {(time.time()-start)/3600:.2f} h -> {fname}")

    def q25(s): return s.quantile(0.25)
    def q75(s): return s.quantile(0.75)
    summary = df.groupby(["topology", "refresh"]).agg(
        median=("state", "median"), q25=("state", q25), q75=("state", q75),
        aligned=("aligned", "median"), modularity=("modularity", "median"),
        gini=("gini_in_degree", "median"), retrains=("retrains", "median"))
    print(summary.round(3))
