#!/usr/bin/env python3
"""Re-run only the pnf x stubbornness heatmap cells whose adaptive horizon moved.

Background
----------
The master heatmap
(Output/heatmap_sweep_phased_sweep_20251014_1511_stubbornness_polarisingNode_f_pxc.csv,
90k rows) predates two independent horizon corrections:

  1. The 45k bug (fixed 2026-07-02, commit 8c8e230). The phased sweep called
     get_adaptive_timesteps(algo, topo) WITHOUT `mode`, so `mode` defaulted to
     "None"; for biased/bridge that builds the key (topology, algo), which is not
     in the factor dict, so the lookup fell through to factor 1.0 and every
     biased/bridge condition ran at a flat 45k regardless of its intended factor.
     -> All 16 biased/bridge conditions are wrong at EVERY stubbornness/pnf.
     Non-biased/bridge conditions legitimately use mode="None", so they were
     unaffected by this bug.
  2. Stubbornness-scaled horizons + six raised base factors (2026-07-03, ea6aa26;
     FB/bridge/diff bumped again 2026-07-06, 2355b9b). See get_adaptive_timesteps
     in general_param_sweep.py.

Because the horizon is a function of (topology, algo, rewiring, stubbornness)
ONLY - never polarisingNode_f - a changed condition means re-running it across
all 10 pnf values, but nothing forces a full-grid redo. This script runs exactly
the affected subset and writes a patch CSV with the same 7 columns as the master;
splice it in with splice_heatmap_rerun.py.

Re-run set (stubbornness = np.linspace(0,1,10), pnf = np.linspace(0,1,10)):

  * FULL group - all rows except s=1 (stubbornness indices 0..8):
      - all 16 biased/bridge conditions            (45k bug: wrong everywhere)
      - Twitter/wtf, cl/random, FB/None            (base factor raised)
  * HIGH group - only s in {0.667, 0.778, 0.889} (indices 6,7,8):
      - every other condition (correct base factor and no 45k bug; only the new
        0.6<s<1 stubbornness scaling moved their horizon)
  * s = 1.0 row is NEVER re-run: agents with s>=1 never update, so the final
    state is horizon-independent and identical to the master's values.

To trim further (optional): if you are confident a biased/bridge condition had
already converged by 45k at low stubbornness, drop those rows from JOBS. The
default here is conservative - these bumps exist precisely because those cells
were under-run, so expect them to move.

Usage:
    cd Analysis && python rerun_subset_sweep.py
"""
import os
import time
import glob
from datetime import datetime

import numpy as np
import pandas as pd

import models_checks
from general_param_sweep import (
    group_scenarios_by_algorithm,
    run_algorithm_phase,
)

MASTER_COLS = ["state", "state_std", "stubbornness",
               "polarisingNode_f", "rewiring", "mode", "topology"]


def build_combined_list():
    """Exactly the scenario list from general_param_sweep.py __main__."""
    rewiring_list_h = ["diff", "same"]
    directed = ["DPAH", "Twitter"]
    undirected = ["cl", "FB"]
    combined_list1 = [(s, r, t) for s in ["biased", "bridge"]
                      for r in rewiring_list_h for t in directed + undirected]
    combined_list2 = [("node2vec", "None", t) for t in directed + undirected]
    combined_list3 = [("None", "None", t) for t in directed + undirected]
    combined_list4 = [("wtf", "None", t) for t in directed]
    combined_list_rand = [("random", "None", t) for t in directed + undirected]
    return (combined_list1 + combined_list_rand + combined_list2
            + combined_list3 + combined_list4)


if __name__ == "__main__":
    numberOfSimulations = 30          # must match the master run
    STUB = np.linspace(0, 1, 10)      # identical floats -> clean splice keys
    PNF = np.linspace(0, 1, 10)

    combined_list = build_combined_list()

    # Conditions whose base horizon changed or that carried the 45k bug: re-run
    # every stubbornness row except s=1 (index 9).
    FULL = ([s for s in combined_list if s[0] in ("biased", "bridge")]
            + [("wtf", "None", "Twitter"),
               ("random", "None", "cl"),
               ("None", "None", "FB")])
    # Everything else: correct base factor, no 45k bug -> only the new stubbornness
    # scaling (0.6 < s < 1) moved the horizon.
    HIGH = [s for s in combined_list if s not in FULL]
    assert len(FULL) == 19 and len(HIGH) == 11, (len(FULL), len(HIGH))

    # (scenarios, stubbornness-index subset). Edit here to trim/extend.
    JOBS = [
        (FULL, list(range(9))),   # s indices 0..8 (all but s=1.0)
        (HIGH, [6, 7, 8]),        # s in {0.667, 0.778, 0.889} only
    ]

    base_args = models_checks.getargs()
    sweep_id = f"rerun_{datetime.now():%Y%m%d_%H%M}"

    total_cells = sum(len(sc) * len(idx) * len(PNF) for sc, idx in JOBS)
    print("=== HEATMAP SUBSET RE-RUN ===")
    print(f"sweep_id: {sweep_id}")
    print(f"cells to re-run: {total_cells} "
          f"(vs {len(combined_list) * len(STUB) * len(PNF)} for the full grid)")
    print(f"sims per cell: {numberOfSimulations} "
          f"-> {total_cells * numberOfSimulations} simulations")
    print("=" * 40)

    results = []
    start = time.time()
    for job_i, (scenarios, stub_idx) in enumerate(JOBS, 1):
        algo_groups = group_scenarios_by_algorithm(scenarios)
        combos = [(si, pnf) for si in stub_idx for pnf in PNF]
        print(f"--- Job {job_i}/{len(JOBS)}: {len(scenarios)} scenarios x "
              f"{len(stub_idx)} stub rows x {len(PNF)} pnf ---")
        for k, (si, pnf) in enumerate(combos, 1):
            params = {"stubbornness": float(STUB[si]),
                      "polarisingNode_f": float(pnf)}
            for _algo_name, algo_scenarios in algo_groups.items():
                results.extend(run_algorithm_phase(
                    algo_scenarios, numberOfSimulations, base_args, params))
            print(f"  [{k}/{len(combos)}] s={STUB[si]:.3f} pnf={pnf:.3f} | "
                  f"{(time.time() - start) / 3600:.2f}h elapsed")

    results_df = pd.DataFrame(results)[MASTER_COLS]
    fname = f"../Output/heatmap_sweep_phased_RERUN_{sweep_id}.csv"
    results_df.to_csv(fname, index=False)

    for f in glob.glob("*embeddings*"):
        os.remove(f)

    print("\n=== RE-RUN COMPLETE ===")
    print(f"runtime: {(time.time() - start) / 3600:.2f} h")
    print(f"patch saved: {fname} ({len(results_df)} rows)")
    print("next: python splice_heatmap_rerun.py "
          f"--master <master.csv> --patch {fname}")
