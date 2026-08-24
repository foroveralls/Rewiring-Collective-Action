# -*- coding: utf-8 -*-
"""
Eq. (3) middle-branch ablation (O3).

The printed Eq. (3) middle branch is

    p1 = 1/(2r) * (x_ij + r + xi)

while the implementation (models_checks.py:247) uses

    p1 = 1/(2r) * (r + xi)

i.e. the code drops the drive term x_ij inside the branch |x_ij + xi| <= r.
Outside that branch the two forms are identical. This script reruns a set of
conditions under BOTH forms and reports whether steady states move.

Design: the two arms are run back to back inside the same worker process, with
the same run index and the same argument dict, so models_checks.simulate()
derives the identical unique_seed (which mixes in os.getpid()) for both. The
network, initial states, friendship weights and the interaction/rewiring
sequence therefore start identical, and the arms differ only through
Agent.consider. The comparison is paired per (condition, run).

Usage:
    cd Analysis
    python eq3_ablation_test.py           # full run
    python eq3_ablation_test.py --smoke   # quick end-to-end validation
"""

import sys
import time
import random
import multiprocessing
from datetime import date
from itertools import repeat

import numpy as np
import pandas as pd
from scipy import stats

import models_checks
from general_param_sweep import get_optimal_process_count, init


# %% the two Agent.consider arms
# Both carry the same branch counter so the middle-branch share is measured
# under identical instrumentation overhead in each arm.
_branch_counts = {"total": 0, "middle": 0}


def _make_consider(printed):
    def consider(self, neighbour, neighboursWeight, politicalClimate):
        self.interactionsReceived += 1
        neighbour.addInteractionGiven()
        if self.stubbornness >= 1:
            return

        mod = 1
        if (self.type == "polarising") & (self.state * neighbour.state < 0):
            mod = -1

        randomness = models_checks.randomness
        weight = (self.state * self.stubbornness + politicalClimate
                  + self.defector_util + neighboursWeight * neighbour.state)

        sample = random.uniform(-randomness, randomness)
        check = weight + sample

        _branch_counts["total"] += 1
        if check < -randomness:
            p1 = 0
        elif check > randomness:
            p1 = 1
        else:
            _branch_counts["middle"] += 1
            if printed:
                p1 = 1 / (2 * randomness) * (weight + randomness + sample)
            else:
                p1 = 1 / (2 * randomness) * (randomness + sample)

        p2 = 1 - p1
        delta = mod * (abs(self.state - neighbour.state)
                       * (p1 * (1 - self.state) - p2 * (1 + self.state)))

        self.state += delta

        if self.state > 1:
            self.state = models_checks.STATES[0]
        elif self.state < -1:
            self.state = models_checks.STATES[1]

    return consider


CONSIDER = {"code": _make_consider(printed=False),
            "printed": _make_consider(printed=True)}
ARMS = ["code", "printed"]

# %% configuration
N_SIMS = 10  # the runs are cheap (< 3 min each); 10 seeds instead of 5 buys power
TOPOLOGIES = ["cl", "DPAH"]          # CSF (undirected) and DPA (directed)
SCENARIOS = [("None", "None"),        # static
             ("random", "None"),      # random rewiring
             ("biased", "same"),      # local (similar)
             ("biased", "diff")]      # local (opposite)

# Canonical horizon for every condition here (bridge/diff/FB run at 270k; none
# of those appear in this ablation).
TIMESTEPS = 160000

# Defaults of the instrumented measurement: r=0.10, w=0.6, phi=0.05, rho=0.10.
POLITICAL_CLIMATE = 0.05
STUBBORNNESS = 0.6
POLARISING_NODE_F = 0.10
NWSIZE = 800


def run_both_arms(i, sim_args):
    """Run one paired (code, printed) replicate inside one process."""
    rows = []
    for arm in ARMS:
        models_checks.Agent.consider = CONSIDER[arm]
        _branch_counts["total"] = 0
        _branch_counts["middle"] = 0
        t0 = time.time()
        sim = models_checks.simulate(i, sim_args)
        states = np.asarray(sim.states)
        tail = states[-len(states) // 10:]      # last 10% of the trajectory
        rows.append({
            "arm": arm, "sim": i,
            "scenario": sim_args["rewiringAlgorithm"],
            "rewiring": sim_args["rewiringMode"],
            "topology": sim_args["type"],
            "timesteps": sim_args["timesteps"],
            "state": states[-1],
            "state_std": sim.statesds[-1],
            "state_tail_mean": tail.mean(),
            "ratio": sim.ratio[-1],
            "middle_share": (_branch_counts["middle"] / _branch_counts["total"]
                             if _branch_counts["total"] else np.nan),
            "runtime_s": time.time() - t0,
        })
    return rows


def run_condition(pool, algo, mode, topo, n_sims, base_args, timesteps):
    sim_args = {**base_args,
                "rewiringAlgorithm": algo, "rewiringMode": mode, "type": topo,
                "top_file": None, "nwsize": NWSIZE, "timesteps": timesteps,
                "politicalClimate": POLITICAL_CLIMATE,
                "newPoliticalClimate": POLITICAL_CLIMATE,
                "stubbornness": STUBBORNNESS,
                "polarisingNode_f": POLARISING_NODE_F,
                "plot": False, "seed": 42}
    batches = pool.starmap(run_both_arms, zip(range(n_sims), repeat(sim_args)))
    return [row for batch in batches for row in batch]


def summarise(df):
    """Paired per-condition comparison of the two arms."""
    keys = ["topology", "scenario", "rewiring"]
    out = []
    for key, g in df.groupby(keys):
        wide = g.pivot(index="sim", columns="arm", values="state")
        diff = wide["printed"] - wide["code"]
        row = dict(zip(keys, key))
        row.update({
            "n": len(diff),
            "code_mean": wide["code"].mean(), "code_sd": wide["code"].std(),
            "printed_mean": wide["printed"].mean(), "printed_sd": wide["printed"].std(),
            "diff_mean": diff.mean(), "diff_sd": diff.std(),
            "middle_share": g["middle_share"].mean(),
        })
        if len(diff) > 1 and diff.std() > 0:
            t, p = stats.ttest_rel(wide["printed"], wide["code"])
            row["t"], row["p"] = t, p
            # Cohen's d on the paired differences
            row["d"] = diff.mean() / diff.std()
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std() / np.sqrt(len(diff))
            row["ci_lo"], row["ci_hi"] = diff.mean() - half, diff.mean() + half
        else:
            row["t"] = row["p"] = row["d"] = row["ci_lo"] = row["ci_hi"] = np.nan
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    n_sims = 2 if smoke else N_SIMS
    timesteps = 2000 if smoke else TIMESTEPS
    if smoke:
        NWSIZE = 200
    nwsize = NWSIZE
    topologies = ["cl"] if smoke else TOPOLOGIES
    scenarios = SCENARIOS[:2] if smoke else SCENARIOS

    conditions = [(algo, mode, topo) for topo in topologies
                  for algo, mode in scenarios]
    base_args = models_checks.getargs()
    n_procs = 2 if smoke else min(get_optimal_process_count(), n_sims)
    total = len(conditions) * len(ARMS) * n_sims

    print(f"=== EQ.(3) MIDDLE-BRANCH ABLATION {'(smoke) ' if smoke else ''}===")
    print(f"Date: {date.today()} | conditions: {len(conditions)} x arms: {len(ARMS)} "
          f"x sims: {n_sims} = {total} runs")
    print(f"N={nwsize}, steps={timesteps}, phi={POLITICAL_CLIMATE}, "
          f"w={STUBBORNNESS}, rho={POLARISING_NODE_F} | processes: {n_procs}")
    print("=" * 40)

    start = time.time()
    results = []
    lock = multiprocessing.Lock()
    with multiprocessing.Pool(processes=n_procs, initializer=init, initargs=(lock,)) as pool:
        for algo, mode, topo in conditions:
            t0 = time.time()
            results.extend(run_condition(pool, algo, mode, topo, n_sims,
                                         base_args, timesteps))
            print(f"  {algo}-{mode} on {topo}: {(time.time()-t0)/60:.1f} min", flush=True)

    df = pd.DataFrame(results)
    suffix = "_smoke" if smoke else ""
    fname = f"../Output/eq3_ablation_{date.today()}{suffix}.csv"
    df.to_csv(fname, index=False)

    summary = summarise(df)
    summary.to_csv(fname.replace(".csv", "_summary.csv"), index=False)
    print(f"\nDone in {(time.time()-start)/3600:.2f} h -> {fname}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(4))
