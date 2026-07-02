# -*- coding: utf-8 -*-
"""
PA ablation test: rerun the heuristic rewiring scenarios with UNIFORM random
partner choice within the candidate pool, replacing the capped degree
preferential selection (select_kmax_new) used in the main results.

Only local ("biased") and bridge rewiring use select_kmax_new
(models_checks.py: biasedrewiring, bridgerewiring); wtf and node2vec have
their own targeting logic and are unaffected by this ablation.

Runs both arms (capped_pa baseline + uniform) so the comparison is
self-contained in one CSV.

Usage:
    cd Analysis
    python pa_ablation_test.py           # full run
    python pa_ablation_test.py --smoke   # quick end-to-end validation
"""

import sys
import time
import random
import multiprocessing
from datetime import date
from itertools import repeat

import numpy as np
import pandas as pd

import models_checks
from general_param_sweep import get_adaptive_timesteps, get_optimal_process_count, init


# %% ablation: uniform partner choice
# The name must match the method it shadows: biasedrewiring/bridgerewiring call
# self.select_kmax_new(pool), and update_instance_methods installs by __name__.
def select_kmax_new(self, node_list, max_prob=0.5):
    """Uniform random partner choice within the candidate pool (PA ablation)."""
    return random.choice(list(node_list))


# update_instance_methods rebinds instance.call_algo to EVERY function passed,
# so the true rewiring algorithm must come last to restore the dispatch.
UNIFORM_FUNC_CHANGES = {
    "biased": [select_kmax_new, models_checks.Model.biasedrewiring],
    "bridge": [select_kmax_new, models_checks.Model.bridgerewiring],
}

# %% configuration
N_SIMS = 20
ALGOS = ["biased", "bridge"]
MODES = ["diff", "same"]
TOPOLOGIES = ["cl", "Twitter"]  # one undirected synthetic, one directed empirical
ARMS = ["capped_pa", "uniform"]

# None -> use the model default (models_checks.args). Set to e.g. 0.10 to match
# the rerun political climate sweep; newPoliticalClimate follows at 1x.
POLITICAL_CLIMATE = None

TOP_FILES = {"Twitter": ("twitter_graph_N_789.gpickle", 789),
             "FB": ("FB_graph_N_786.gpickle", 786)}


def _simulate_checked(i, newArgs, func_changes):
    """Worker-side wrapper: verify the ablation is installed before returning.
    Bound methods pickle as getattr(obj, name) and so revert to the class
    attribute on the round trip back to the parent, which means this check
    cannot be done on the returned models."""
    model = models_checks.simulate(i, newArgs, func_changes)
    if func_changes:
        assert model.select_kmax_new.__func__ is select_kmax_new, \
            "uniform selection was not installed on the model instance"
    return model


def run_scenario(pool, arm, algo, mode, topo, n_sims, base_args, smoke=False):
    top_file, nwsize = TOP_FILES.get(topo, (None, 800))
    timesteps = get_adaptive_timesteps(algo, topo, mode)
    if smoke:
        nwsize, timesteps = 200, 1000

    sim_args = {**base_args,
                "rewiringAlgorithm": algo, "rewiringMode": mode, "type": topo,
                "top_file": top_file, "nwsize": nwsize, "timesteps": timesteps,
                "plot": False, "seed": 42}
    if POLITICAL_CLIMATE is not None:
        sim_args["politicalClimate"] = POLITICAL_CLIMATE
        sim_args["newPoliticalClimate"] = POLITICAL_CLIMATE

    func_changes = UNIFORM_FUNC_CHANGES[algo] if arm == "uniform" else False
    sims = pool.starmap(_simulate_checked,
                        zip(range(n_sims), repeat(sim_args), repeat(func_changes)))

    algos_run = {str(m.algo) for m in sims}
    if len(algos_run) > 1:
        raise ValueError(f"Mixed algorithms in batch: {algos_run}")

    rows = []
    for i, sim in enumerate(sims):
        degrees = np.array([d for _, d in sim.graph.degree()])
        rows.append({
            "selection": arm, "scenario": algo, "rewiring": mode,
            "topology": topo, "sim": i, "timesteps": timesteps,
            "state": sim.states[-1], "state_std": sim.statesds[-1],
            "avg_degree": degrees.mean(), "sd_degree": degrees.std(),
            "max_degree": degrees.max(), "min_degree": degrees.min(),
        })
    return rows


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    n_sims = 2 if smoke else N_SIMS
    modes = ["diff"] if smoke else MODES
    topologies = ["cl"] if smoke else TOPOLOGIES

    scenarios = [(algo, mode, topo) for algo in ALGOS
                 for mode in modes for topo in topologies]
    base_args = models_checks.getargs()

    n_procs = 2 if smoke else get_optimal_process_count()
    total = len(scenarios) * len(ARMS) * n_sims
    climate = POLITICAL_CLIMATE if POLITICAL_CLIMATE is not None else base_args["politicalClimate"]

    print(f"=== PA ABLATION {'(smoke) ' if smoke else ''}===")
    print(f"Date: {date.today()} | scenarios: {len(scenarios)} x arms: {len(ARMS)} x sims: {n_sims} = {total} runs")
    print(f"Political climate: {climate} | processes: {n_procs}")
    print("=" * 40)

    start = time.time()
    results = []
    lock = multiprocessing.Lock()
    with multiprocessing.Pool(processes=n_procs, initializer=init, initargs=(lock,)) as pool:
        for algo, mode, topo in scenarios:
            for arm in ARMS:
                t0 = time.time()
                results.extend(run_scenario(pool, arm, algo, mode, topo,
                                            n_sims, base_args, smoke=smoke))
                print(f"  {algo}-{mode} on {topo} [{arm}]: {(time.time()-t0)/60:.1f} min")

    df = pd.DataFrame(results)
    suffix = "_smoke" if smoke else ""
    fname = f"../Output/pa_ablation_{date.today()}{suffix}.csv"
    df.to_csv(fname, index=False)
    print(f"\nDone in {(time.time()-start)/3600:.2f} h -> {fname}")

    summary = df.groupby(["scenario", "rewiring", "topology", "selection"])[
        ["state", "state_std", "max_degree"]].mean()
    print(summary.round(3))
