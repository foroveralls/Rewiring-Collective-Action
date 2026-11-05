#!/usr/bin/env python3
"""
LOW-MEMORY extractor for PER-RUN network metrics from the snapshot pickle.

Writes one row per (run, snapshot) so SI Fig. S2 can carry IQR bands and per-run
hairlines (Analysis/Plotting/network_metrics_bands.py); per-run opinion metrics come
from Output/per_run_summary.csv instead.

HOW: same trick as extract_states_lowmem.py -- a custom Unpickler swaps the graph
classes for a stub. The difference is that metrics NEED the edges, so the stub does
not discard them: it rebuilds a plain networkx graph over the same adjacency dicts
(no copy), computes the four metrics, and keeps only the resulting floats. Each
graph's adjacency is therefore live for one graph at a time rather than all 6750.

Peak memory is still dominated by the unpickler memo, exactly as for the per-agent
extraction, so the full campaign pickle needs a large-memory machine. A small pickle
from a single-condition run is enough for testing (see --pkl below).

METRIC CONVENTIONS:

  * Graphs are measured AS STORED. DPAH and Twitter are directed, CSF (PATCH) and
    FB are undirected; directed graphs get the directed clustering and
    assortativity, not those of an undirected projection (on a test DPAH,
    clustering 0.27 directed vs 0.41 undirected).
  * Conversion is to nx.DiGraph / nx.Graph rather than left as the netin subclass,
    because louvain_communities reconstructs the graph class and netin generators
    require constructor arguments (TypeError otherwise). Direction is preserved.
  * gini_in_degree uses in-degree for directed graphs and degree for undirected
    ones, where the two coincide by definition.
  * Communities are detected with LEIDEN, unweighted, seed=42 -- the same method and
    settings the model itself uses (Traag et al. 2019). On FB at t=0 the alternatives
    give Q = 0.5773 (Leiden), 0.5759 (Louvain), 0.5205 (greedy modularity, CNM);
    --community selects either of the others.
    NOTE: the model computes its partition ONCE at initialisation and caches it
    because bridge targets are defined against initial communities.
    This script recomputes the partition at every snapshot on purpose: the figure
    reports how community structure itself evolves, which is a different quantity
    from the cached rewiring partition.

Run from Analysis/ with the conda python:
  cd Analysis && python \
      extract_network_metrics_lowmem.py \
      --pkl ../Output/all_snapshots_sweep_20251014_1704_phased_run_gme_<date>.pkl.gz

Local smoke test (47 MB, one combo, loads in ~3 s):
  ... extract_network_metrics_lowmem.py \
      --pkl ../Output/all_snapshots_<single-condition run>.pkl.gz \
      --out ../Output/network_metrics_per_run_SMOKE.csv

Cost: ~0.38 s per graph measured on a real 786-node/14k-edge FB snapshot, so the
full 30 combos x 45 runs x 5 timesteps = 6750 graphs takes roughly 45 min serial.
"""
import os, sys, gc, gzip, pickle, csv, time, argparse

import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_community
import igraph as ig
import leidenalg as la

# Derived, not hardcoded: this script also runs on the cluster.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "..", "Output")
sys.path.insert(0, ANALYSIS_DIR)

_ap = argparse.ArgumentParser(description="Extract per-run network metrics from a master snapshot pickle.")
_ap.add_argument("--pkl", required=True,
                 help="master snapshot pickle to read (no default on purpose: the "
                      "figure must be campaign-consistent with Figs 2/3, i.e. gme)")
_ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "network_metrics_per_run.csv"),
                 help="per-run metric CSV to write; the _META.txt sidecar follows its name")
_ap.add_argument("--community", default="leiden", choices=["leiden", "louvain", "greedy"],
                 help="community detection for modularity. 'leiden' (default) matches the "
                      "model; 'louvain' and 'greedy' are the networkx alternatives")
_ap.add_argument("--weight", default="none", choices=["none", "weight"],
                 help="edge attribute for modularity ('none' = unweighted, default, and "
                      "what the model's own Leiden call does)")
_args = _ap.parse_args()

PKL = os.path.abspath(_args.pkl)
OUT_CSV = os.path.abspath(_args.out)
META_TXT = os.path.splitext(OUT_CSV)[0] + "_META.txt"
COMMUNITY = _args.community
WEIGHT = None if _args.weight == "none" else _args.weight

os.chdir(ANALYSIS_DIR)

t0 = time.time()
import models_checks  # noqa: resolves the real Agent class referenced by node attrs

GRAPH_NAMES = {"Graph", "DiGraph", "MultiGraph", "MultiDiGraph",
               "DPAH", "PATCH", "DPA", "DH", "PA", "PAH"}

METRIC_FIELDS = ["directed", "n_nodes", "n_edges",
                 "clustering", "modularity", "assortativity", "gini_in_degree"]

_counter = {"graphs": 0}


def _gini(values):
    """Gini coefficient, O(n log n). Equivalent to the old O(n^2) pairwise form."""
    x = np.sort(np.asarray(list(values), dtype=float))
    n = x.size
    if n == 0:
        return np.nan
    total = x.sum()
    if total <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def _modularity(g):
    """Modularity of the instantaneous community structure.

    Leiden goes through igraph. The graph is rebuilt from the edge list rather than
    ig.Graph.from_networkx() so that igraph does not copy the per-node Agent objects,
    which is both slow and pointless here. Direction is preserved, so directed graphs
    get the directed modularity.
    """
    if COMMUNITY == 'leiden':
        nodes = list(g.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in g.edges()]
        ig_g = ig.Graph(n=len(nodes), edges=edges, directed=g.is_directed())
        kwargs = {}
        if WEIGHT is not None:
            kwargs['weights'] = [d.get(WEIGHT, 1.0) for _u, _v, d in g.edges(data=True)]
        return la.find_partition(ig_g, la.ModularityVertexPartition,
                                 seed=42, **kwargs).modularity

    if COMMUNITY == 'louvain':
        comms = nx_community.louvain_communities(g, weight=WEIGHT, seed=42)
    else:
        comms = nx_community.greedy_modularity_communities(g, weight=WEIGHT)
    return nx_community.modularity(g, comms, weight=WEIGHT)


def _metrics_from_state(state):
    """Rebuild a plain networkx graph over the unpickled adjacency and measure it.

    The adjacency dicts are reused, not copied, so this adds no meaningful memory
    on top of what the unpickler already holds for this one graph.
    """
    directed = "_pred" in state or "_succ" in state
    g = nx.DiGraph() if directed else nx.Graph()
    for k in ("graph", "_node", "_adj", "_succ", "_pred"):
        if k in state:
            g.__dict__[k] = state[k]
    if directed and "_adj" not in state and "_succ" in state:
        g.__dict__["_adj"] = state["_succ"]

    out = {"directed": int(directed),
           "n_nodes": g.number_of_nodes(),
           "n_edges": g.number_of_edges()}

    if out["n_nodes"] == 0 or out["n_edges"] == 0:
        out.update({k: np.nan for k in ("clustering", "modularity",
                                        "assortativity", "gini_in_degree")})
        return out

    # Directed graphs get the directed clustering coefficient; undirected the usual one.
    try:
        out["clustering"] = nx.average_clustering(g)
    except Exception:
        out["clustering"] = np.nan

    try:
        out["modularity"] = _modularity(g)
    except Exception:
        out["modularity"] = np.nan

    # networkx defaults to out-in assortativity on a DiGraph, the standard directed
    # analogue of the undirected degree-degree correlation.
    try:
        out["assortativity"] = nx.degree_assortativity_coefficient(g)
    except Exception:
        out["assortativity"] = np.nan

    # In-degree for directed graphs; for undirected ones in-degree IS degree.
    degrees = (d for _n, d in (g.in_degree() if directed else g.degree()))
    out["gini_in_degree"] = _gini(degrees)

    return out


class _MetricGraph:
    """Stand-in for a graph: on unpickle, keep only its measured metrics."""
    __slots__ = ("metrics",)

    def __setstate__(self, state):
        if not isinstance(state, dict):
            self.metrics = None
            return
        self.metrics = _metrics_from_state(state)
        _counter["graphs"] += 1
        if _counter["graphs"] % 250 == 0:
            print(f"[{time.time()-t0:7.1f}s] measured {_counter['graphs']} graphs", flush=True)
        # state (with _adj etc.) is dropped when this returns


class MetricOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in GRAPH_NAMES and ("networkx" in module or "netin" in module):
            return _MetricGraph
        return super().find_class(module, name)


print(f"[{time.time()-t0:7.1f}s] streaming + measuring {PKL} ...", flush=True)
print(f"[{time.time()-t0:7.1f}s] community = {COMMUNITY}, weight = {WEIGHT!r}", flush=True)
with gzip.open(PKL, "rb") as f:
    master = MetricOnlyUnpickler(f).load()
print(f"[{time.time()-t0:7.1f}s] done. combos={len(master)} graphs={_counter['graphs']}", flush=True)

meta_lines = [f"source pickle: {PKL}",
              f"community detection: {COMMUNITY} (weight={WEIGHT!r})",
              f"combos ({len(master)}): {list(master.keys())}"]
rows = 0
with open(OUT_CSV, "w", newline="") as fout:
    w = csv.writer(fout)
    w.writerow(["scenario", "rewiring", "type", "model_run", "timestep"] + METRIC_FIELDS)
    for key in list(master.keys()):
        entry = master[key]
        meta = entry.get("metadata", {})
        algo, mode, topo = meta.get("algo"), meta.get("mode"), meta.get("topology")
        params = meta.get("params", {})
        snaps = entry.get("snapshots", {})
        run_ids = sorted(snaps.keys())
        ts_keys = sorted(snaps[run_ids[0]].keys()) if run_ids else []
        line = (f"{key}: pNf={params.get('polarisingNode_f')} "
                f"nwsize={params.get('nwsize')} timesteps={params.get('timesteps')} "
                f"n_runs={len(run_ids)} ts_keys={ts_keys}")
        meta_lines.append(line)
        print(f"[{time.time()-t0:7.1f}s] {line}", flush=True)
        for run in run_ids:
            for t, g in sorted(snaps[run].items()):
                m = getattr(g, "metrics", None)
                if m is None:
                    continue
                w.writerow([algo, mode, topo, run, t] + [m[f] for f in METRIC_FIELDS])
                rows += 1
        del master[key]
        gc.collect()

meta_lines.append(f"TOTAL rows: {rows}")
with open(META_TXT, "w") as mf:
    mf.write("\n".join(meta_lines) + "\n")
print(f"[{time.time()-t0:7.1f}s] DONE rows={rows} -> {OUT_CSV}", flush=True)
