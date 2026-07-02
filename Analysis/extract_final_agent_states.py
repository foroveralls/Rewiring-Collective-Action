#!/usr/bin/env python3
"""
Extract per-agent FINAL opinion states from the consolidated snapshot pickle.

The snapshot pickle stores, per scenario_key, a dict:
    {'metadata': {...}, 'snapshots': {run_index: {timestep: graph}}}
where each graph node carries nodes[n]['agent'].state (the agent opinion).

We pull only the final-timestep graph for every available run and write a
compact long CSV with one row per (combo, run, agent). This is the data needed
for a *proper* opinion-distribution violin (every agent in the ensemble), which
the avg/individual CSVs (which store only per-run avg_state) cannot provide.

Run from the Analysis/ directory so `import models_checks` resolves the pickled
Agent class.
"""
import os, sys, gc, gzip, pickle, csv, time

ANALYSIS_DIR = "/home/jpoveralls/Documents/Projects_code/Rewiring-Collective-Action/Analysis"
sys.path.insert(0, ANALYSIS_DIR)
os.chdir(ANALYSIS_DIR)

PKL = "../Output/all_snapshots_sweep_20250905_1501_phased_run_afk_2025-09-05.pkl.gz"
OUT_CSV = "../Output/per_agent_final_states.csv"
META_TXT = "../Output/per_agent_final_states_META.txt"

t0 = time.time()
print(f"[{time.time()-t0:6.1f}s] importing models_checks ...", flush=True)
import models_checks  # noqa: needed so unpickler can resolve Agent etc.

print(f"[{time.time()-t0:6.1f}s] loading pickle (this is the heavy step) ...", flush=True)
with gzip.open(PKL, "rb") as f:
    master = pickle.load(f)
print(f"[{time.time()-t0:6.1f}s] loaded. top-level combos: {len(master)}", flush=True)

combos = list(master.keys())
meta_lines = []
meta_lines.append(f"combos ({len(combos)}): {combos}")

def get_state(node_data):
    a = node_data.get("agent", None)
    if a is not None and hasattr(a, "state"):
        return a.state
    return node_data.get("state", None)

rows_written = 0
with open(OUT_CSV, "w", newline="") as fout:
    w = csv.writer(fout)
    w.writerow(["scenario", "rewiring", "type", "model_run", "agent", "final_t", "state"])

    for key in combos:
        entry = master[key]
        meta = entry.get("metadata", {})
        algo = meta.get("algo")
        mode = meta.get("mode")
        topo = meta.get("topology")
        params = meta.get("params", {})
        pNf = params.get("polarisingNode_f")
        nwsize = params.get("nwsize")
        tsteps = params.get("timesteps")
        snaps = entry.get("snapshots", {})
        run_ids = sorted(snaps.keys())

        # inspect timestep keys from the first run
        ts_keys = sorted(snaps[run_ids[0]].keys()) if run_ids else []
        final_t = ts_keys[-1] if ts_keys else None

        line = (f"{key}: algo={algo} mode={mode} topo={topo} "
                f"pNf={pNf} nwsize={nwsize} timesteps={tsteps} "
                f"n_runs={len(run_ids)} run_ids={run_ids[:3]}..{run_ids[-1:]} "
                f"ts_keys={ts_keys} final_t={final_t}")
        meta_lines.append(line)
        print(f"[{time.time()-t0:6.1f}s] {line}", flush=True)

        for run in run_ids:
            tmap = snaps[run]
            ft = max(tmap.keys())
            g = tmap[ft]
            for n in g.nodes():
                st = get_state(g.nodes[n])
                if st is None:
                    continue
                w.writerow([algo, mode, topo, run, n, ft, st])
                rows_written += 1

        # free this combo before moving on
        del master[key]
        del entry, snaps
        gc.collect()

meta_lines.append(f"TOTAL rows written: {rows_written}")
with open(META_TXT, "w") as mf:
    mf.write("\n".join(meta_lines) + "\n")

print(f"[{time.time()-t0:6.1f}s] DONE. rows={rows_written} -> {OUT_CSV}", flush=True)
print(f"meta -> {META_TXT}", flush=True)
