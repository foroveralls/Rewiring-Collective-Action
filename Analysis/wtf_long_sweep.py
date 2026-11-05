#!/usr/bin/env python3
"""WTF cells of the two sensitivity sweeps at the long WTF horizon.

WTF runs switch between opposed and aligned states, so the WTF cells of the sweeps run longer than
the horizons general_param_sweep.py uses (45k x a per-condition factor). The paper's
WTF sweep values come from this script, at 500k steps, scaled for the (rho, w) grid by the sweep's
stubbornness factor (640k / 1.1M / 2.8M at w = 0.67 / 0.78 / 0.89). Its rows replace the WTF rows of
general_param_sweep.py's output (same param1 / param2 / rewiring / mode / topology / run columns).

Grids as in general_param_sweep.py, DPAH and Twitter, 30 runs per cell:
  rhow: stubbornness x polarisingNode_f, linspace(0, 1, 10) each. w = 1 is skipped: agents with
        stubbornness 1 never update, so that row is the same at any horizon.
  phi:  politicalClimate linspace(0, 0.05, 12) at polarisingNode_f = 0.10.
Settings: DPAH (rho, w) cells at w < 0.6 run with NETWORK_CACHE=0 and WEIGHT_BATCH=1 (see
models_checks.py), all other cells with the defaults; the script sets both per run.

Per run: last step and last-10% mean of the mean opinion (state) and of polarization (state_std), and
the mean-opinion trajectory every 1,000 steps in a second file. Rows are appended as runs finish, so a restart skips completed runs.

Usage, from Analysis/: python wtf_long_sweep.py <rhow|phi> [workers]
"""
import csv
import multiprocessing
import os
import sys
from itertools import product

import numpy as np

import models_checks as mc
from general_param_sweep import STUB_CAP, STUB_EXP, STUB_REF
from run_phased import get_optimal_process_count

BASE = 500_000
EVERY = 1000
N_RUNS = 30
TOPS = {"DPAH": (None, 800), "Twitter": ("twitter_graph_N_789.gpickle", 789)}
FIELDS = ["state", "state_last10", "state_std", "state_std_last10", "param1", "param2",
          "rewiring", "mode", "topology", "run", "timesteps", "network_cache"]


def cells(grid):
    if grid == "rhow":
        g = [float(x) for x in np.linspace(0, 1, 10)]
        return [{"stubbornness": s, "polarisingNode_f": f} for s, f in product(g, g) if s < 1]
    return [{"politicalClimate": float(p), "polarisingNode_f": 0.10} for p in np.linspace(0, 0.05, 12)]


def horizon(grid, s):
    if grid == "rhow" and STUB_REF < s < 1:
        return int(BASE * min(STUB_CAP, ((1 - STUB_REF) / (1 - s)) ** STUB_EXP))
    return BASE


def network_cache(grid, topo, s):
    return 0 if grid == "rhow" and topo == "DPAH" and s < 0.6 else 1


def run(task):
    grid, params, topo, i, steps, cache = task
    mc.USE_NETWORK_CACHE = bool(cache)  # read at call time by _cached_network
    mc.WEIGHT_BATCH = 4096 if cache else 1
    top_file, nwsize = TOPS[topo]
    a = {**mc.getargs(), "rewiringAlgorithm": "wtf", "rewiringMode": "None", "type": topo,
         "top_file": top_file, "nwsize": nwsize, "timesteps": steps, "plot": False, "seed": 42, **params}
    m = mc.simulate(i, a)
    assert m.wtf_attempts == 1, m.wtf_attempts
    s, sd = np.asarray(m.states), np.asarray(m.statesds)
    p1 = params["stubbornness"] if grid == "rhow" else params["politicalClimate"]
    row = [s[-1], s[-steps // 10:].mean(), sd[-1], sd[-steps // 10:].mean(), p1, params["polarisingNode_f"],
           "None", "wtf", topo, i, steps, cache]
    return row, s[::EVERY].tolist()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("rhow", "phi"):
        raise SystemExit(__doc__)
    grid = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else get_optimal_process_count()
    assert mc.wtf_attempts == 1, mc.wtf_attempts
    out, traj = f"../Output/wtf_long_sweep_{grid}.csv", f"../Output/wtf_long_sweep_{grid}_traj.csv"
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["param1"], r["param2"], r["topology"], r["run"]) for r in csv.DictReader(f)}
    tasks = []
    for c in cells(grid):
        s = c.get("stubbornness", 0.0)
        p1 = c["stubbornness"] if grid == "rhow" else c["politicalClimate"]
        for t in TOPS:
            tasks += [(grid, c, t, i, horizon(grid, s), network_cache(grid, t, s)) for i in range(N_RUNS)
                      if (str(p1), str(c["polarisingNode_f"]), t, str(i)) not in done]
    tasks.sort(key=lambda x: (-x[4], x[2] != "DPAH"))  # longest runs first, so the tail of the job is short
    print(f"{grid}: {len(tasks)} runs to do, {len(done)} done, {sum(x[4] for x in tasks):.3g} steps, "
          f"{workers} workers -> {out}", flush=True)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f, open(traj, "a", newline="") as g, multiprocessing.Pool(workers) as pool:
        w, wt = csv.writer(f), csv.writer(g)
        if new:
            w.writerow(FIELDS)
            wt.writerow(["param1", "param2", "topology", "run", "t", "a"])
        for k, (row, tr) in enumerate(pool.imap_unordered(run, tasks), 1):
            # trajectory first, so a row in `out` (the restart key) implies its trajectory is on disk
            wt.writerows([row[4], row[5], row[8], row[9], j * EVERY, x] for j, x in enumerate(tr))
            g.flush()
            w.writerow(row)
            f.flush()
            if k % 100 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)}", flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    main()
