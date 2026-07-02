#!/usr/bin/env python3
"""
LOW-MEMORY extractor for per-agent FINAL opinion states from the snapshot pickle.

WHY: the naive `extract_final_agent_states.py` calls pickle.load() on the whole
1.2 GB gzipped master pickle, which OOM-kills on a ~15 GB / ~6-11 GB-free box.
The memory bulk is EDGES (_adj/_succ/_pred) across ~6750 saved graphs. We don't
need edges for an opinion violin -- only nodes[n]['agent'].state.

HOW: a custom Unpickler replaces the graph classes (netin DPAH/PATCH and
networkx Graph/DiGraph) with a stub whose __setstate__ extracts the per-node
agent.state immediately and discards _node/_adj/etc. So each graph's edges are
built only transiently (one graph at a time) and freed before the next, keeping
peak memory low. The real Agent class is kept so .state is readable.

STATUS: DRAFT - NOT YET RUN/VERIFIED. Risk: if a netin generator defines a
custom __reduce__/__setstate__ that regenerates the graph instead of restoring
__dict__, this stub approach needs adjustment. Try it; if it errors on a class,
add that class to GRAPH_NAMES handling or inspect its reduce.

If this still struggles, set SUBSET_RUNS to e.g. 10 to cap runs per combo
(NOTE: cap currently cannot reduce the unpickle cost because pickle streams the
whole object; the cap only limits what we keep -- memory is already low here).

Run from Analysis/ with the conda python:
  cd Analysis && /home/jpoveralls/miniconda3/envs/collective_rewiring/bin/python \
      extract_states_lowmem.py
"""
import os, sys, gc, gzip, pickle, csv, time, io

ANALYSIS_DIR = "/home/jpoveralls/Documents/Projects_code/Rewiring-Collective-Action/Analysis"
sys.path.insert(0, ANALYSIS_DIR)
os.chdir(ANALYSIS_DIR)

PKL = "../Output/all_snapshots_sweep_20250905_1501_phased_run_afk_2025-09-05.pkl.gz"
OUT_CSV = "../Output/per_agent_final_states.csv"
META_TXT = "../Output/per_agent_final_states_META.txt"

t0 = time.time()
import models_checks  # noqa: resolves the real Agent class

# Graph classes to stub (anything that is a networkx/netin graph container).
GRAPH_NAMES = {"Graph", "DiGraph", "MultiGraph", "MultiDiGraph",
               "DPAH", "PATCH", "DPA", "DH", "PA", "PAH"}


class _StubGraph:
    """Stand-in for a graph: on unpickle, keep only the list of agent states."""
    __slots__ = ("agent_states",)

    def __setstate__(self, state):
        node = state.get("_node", {}) if isinstance(state, dict) else {}
        sts = []
        for _n, attr in node.items():
            a = attr.get("agent") if isinstance(attr, dict) else None
            if a is not None and hasattr(a, "state"):
                sts.append(float(a.state))
            elif isinstance(attr, dict) and "state" in attr:
                sts.append(float(attr["state"]))
        self.agent_states = sts
        # state dict (with _adj etc.) is dropped when this returns


class StateOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in GRAPH_NAMES and ("networkx" in module or "netin" in module):
            return _StubGraph
        return super().find_class(module, name)


print(f"[{time.time()-t0:6.1f}s] streaming + stub-unpickling {PKL} ...", flush=True)
with gzip.open(PKL, "rb") as f:
    master = StateOnlyUnpickler(f).load()
print(f"[{time.time()-t0:6.1f}s] done. combos={len(master)}", flush=True)

meta_lines = [f"combos ({len(master)}): {list(master.keys())}"]
rows = 0
with open(OUT_CSV, "w", newline="") as fout:
    w = csv.writer(fout)
    w.writerow(["scenario", "rewiring", "type", "model_run", "final_t", "state"])
    for key in list(master.keys()):
        entry = master[key]
        meta = entry.get("metadata", {})
        algo, mode, topo = meta.get("algo"), meta.get("mode"), meta.get("topology")
        params = meta.get("params", {})
        snaps = entry.get("snapshots", {})
        run_ids = sorted(snaps.keys())
        ts_keys = sorted(snaps[run_ids[0]].keys()) if run_ids else []
        final_t = ts_keys[-1] if ts_keys else None
        line = (f"{key}: pNf={params.get('polarisingNode_f')} "
                f"nwsize={params.get('nwsize')} timesteps={params.get('timesteps')} "
                f"n_runs={len(run_ids)} ts_keys={ts_keys} final_t={final_t}")
        meta_lines.append(line)
        print(f"[{time.time()-t0:6.1f}s] {line}", flush=True)
        for run in run_ids:
            tmap = snaps[run]
            ft = max(tmap.keys())
            g = tmap[ft]
            for s in getattr(g, "agent_states", []):
                w.writerow([algo, mode, topo, run, ft, s])
                rows += 1
        del master[key]
        gc.collect()

meta_lines.append(f"TOTAL rows: {rows}")
with open(META_TXT, "w") as mf:
    mf.write("\n".join(meta_lines) + "\n")
print(f"[{time.time()-t0:6.1f}s] DONE rows={rows} -> {OUT_CSV}", flush=True)
