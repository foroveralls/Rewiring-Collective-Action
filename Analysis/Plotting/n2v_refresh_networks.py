"""
Exploratory: how the Fig. 2(b) example network evolves under node2vec when the
embedding is retrained on the live graph every R steps (n2v_refresh = R), against
the canonical static embedding. Not a paper figure.

Same parameters, run seeds, edge-frequency averaging and drawing as
plot_transformation_circle in transformation_grid_plot.py. The initial network is
taken from the published panel's pickle (Figs/Networks/transformation_circle_N100_
DPAH_n1_thresh0.1_20251021_networks.pkl: one run, final graph at 20,000 steps);
regenerating it with today's code gives a different graph (245 vs 237 edges).
At N = 100 one sweep is 100 steps, so R = 1000 and R = 100 are the analogues of the
pilot's R = 8000 and R = 800 at N = 800 (n2v_refresh_pilot.py).

Usage:
    cd Analysis/Plotting
    python n2v_refresh_networks.py [--runs 30] [--steps 20000] [--refresh 0 1000 100]
                                   [--init-pickle PATH | --regenerate]
"""
import os
import sys
import random
import pickle
import argparse
import multiprocessing
from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append('..')
import models_checks
from general_param_sweep import get_optimal_process_count, init
from n2v_refresh_pilot import _modularity, _gini, _in_degrees
from transformation_grid_plot import (set_plot_style, create_avg_network,
                                      plot_network_compact, FONT_SIZE, cm)

# plot_transformation_circle's base parameters, node2vec only
BASE_PARAMS = {
    "type": "DPAH", "nwsize": 100, "degree": 2, "polarisingNode_f": 0.10,
    "plot": False, "friendship": 0.5, "friendshipSD": 0.19, "skew": -0.25,
    "initSD": 0.15, "stubbornness": 0.6, "politicalClimate": 0.05,
    "newPoliticalClimate": 0.05, "randomness": 0.10, "continuous": True,
    "clustering": 0.5, "defectorUtility": 0.0, "wtf_freq": 10, "breaklinkprob": 1,
    "establishlinkprob": 0.5, "seed": 42, "f_all": 0.5, "save_snapshots": False,
    "rewiringAlgorithm": "node2vec", "rewiringMode": "None",
}
THRESHOLD = 0.1  # edge-frequency threshold of the published panel
PUBLISHED_INIT = "../../Figs/Networks/transformation_circle_N100_DPAH_n1_thresh0.1_20251021_networks.pkl"


def _generators(p):
    return (models_checks.get_truncated_normal(p["friendship"], p["friendshipSD"], 0, 1),
            models_checks.get_truncated_normal(p["skew"], p["initSD"], -1, 1))


def frozen_init_graph(p):
    """The Fig. 2(b) initial network: same seed and call order as plot_transformation_circle."""
    from netin import DPAH
    np.random.seed(42)
    random.seed(42)
    gen = DPAH(p["nwsize"], f_m=0.5, d=0.02, h_MM=p["f_all"], h_mm=p["f_all"],
               plo_M=2.0, plo_m=2.0, seed=42)
    gen.generate()
    fw, init_state = _generators(p)
    temp = models_checks.Model(friendshipWeightGenerator=fw, initialStateGenerator=init_state,
                               args={**p, "rewiringAlgorithm": "None"})
    temp.graph = gen
    temp.populateModel_netin(p["nwsize"], p["skew"])
    g = nx.DiGraph()
    g.add_nodes_from(temp.graph.nodes(data=True))
    g.add_edges_from(temp.graph.edges(data=True))
    return g


def _run(i, p, init_graph):
    """One run from the frozen network, seeded as run_sims seeds run i. Returns
    plain data; the model never goes back through the pool."""
    seed = 42 + 2 * i * 1000
    np.random.seed(seed)
    random.seed(seed)
    fw, init_state = _generators(p)
    model = models_checks.Model(friendshipWeightGenerator=fw, initialStateGenerator=init_state,
                                args=p)
    model.graph = deepcopy(init_graph)
    model.snapshots = {}
    model.runSim(p["timesteps"], clusters=True, drawModel=False, save_snapshots=False)
    g = model.graph
    init_edges = set(init_graph.edges())
    return {
        "states": {n: g.nodes[n]["agent"].state for n in g.nodes()},
        "edges": list(g.edges()),
        "row": {"refresh": p["n2v_refresh"], "sim": i, "state": model.states[-1],
                "aligned": model.ratio[-1], "modularity": _modularity(g),
                "gini_in_degree": _gini(_in_degrees(g)),
                "original_edges_kept": len(init_edges & set(g.edges())) / len(init_edges),
                "retrains": model.retrain},
    }


def _as_model(result):
    """Minimal stand-in that create_avg_network can read (graph with agent states)."""
    g = nx.DiGraph()
    g.add_nodes_from((n, {"agent": SimpleNamespace(state=s)}) for n, s in result["states"].items())
    g.add_edges_from(result["edges"])
    return SimpleNamespace(snapshots={}, graph=g)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--init-pickle", default=PUBLISHED_INIT,
                    help="networks.pkl from plot_transformation_circle; its frozen_init_graph is used")
    ap.add_argument("--regenerate", action="store_true",
                    help="build the initial network with the current code instead")
    ap.add_argument("--refresh", type=int, nargs="+", default=[0, 1000, 100])
    a = ap.parse_args()

    p = {**BASE_PARAMS, "timesteps": a.steps}
    if a.regenerate:
        init_graph = frozen_init_graph(p)
    else:
        with open(a.init_pickle, "rb") as f:
            init_graph = pickle.load(f)["frozen_init_graph"]
    print(f"Frozen DPAH: {init_graph.number_of_nodes()} nodes, {init_graph.number_of_edges()} edges")

    jobs = [(i, {**p, "n2v_refresh": R}, init_graph) for R in a.refresh for i in range(a.runs)]
    n_procs = min(len(jobs), get_optimal_process_count())
    print(f"{len(jobs)} runs ({a.runs} x refresh {a.refresh}), {a.steps} steps, {n_procs} processes")
    with multiprocessing.Pool(n_procs, initializer=init, initargs=(multiprocessing.Lock(),)) as pool:
        results = pool.starmap(_run, jobs)

    df = pd.DataFrame([r["row"] for r in results])
    summary = df.groupby("refresh").median(numeric_only=True).drop(columns="sim")
    pd.set_option("display.width", 200)
    print("\nMedians over runs:")
    print(summary.round(3))

    set_plot_style()
    fig, axes = plt.subplots(1, len(a.refresh) + 1, figsize=(6 * cm * (len(a.refresh) + 1), 6.5 * cm))
    g0 = nx.DiGraph()
    g0.add_nodes_from((n, {"avg_opinion": init_graph.nodes[n]["agent"].state}) for n in init_graph)
    g0.add_edges_from(init_graph.edges(), weight=1.0, rewired=False)
    plot_network_compact(g0, axes[0], threshold=THRESHOLD)
    axes[0].set_title("Initial (t=0)", fontsize=FONT_SIZE, weight="bold")
    for ax, R in zip(axes[1:], a.refresh):
        models = [_as_model(r) for r in results if r["row"]["refresh"] == R]
        G = create_avg_network(models, a.steps - 1, threshold=THRESHOLD, init_graph=init_graph)
        plot_network_compact(G, ax, threshold=THRESHOLD, algorithm="node2vec")
        s = summary.loc[R]
        ax.set_title("N2V static" if R == 0 else f"N2V R={R}", fontsize=FONT_SIZE)
        ax.set_xlabel(f"Q={s.modularity:.2f}  Gini={s.gini_in_degree:.2f}\n"
                      f"kept={s.original_edges_kept:.2f}  a*={s.state:.2f}", fontsize=FONT_SIZE - 2)

    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Networks", exist_ok=True)
    stem = f"../../Figs/Networks/n2v_refresh_networks_N100_DPAH_n{a.runs}_{today}"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    df.to_csv(f"../../Output/n2v_refresh_networks_N100_{today}.csv", index=False)
    print(f"\nSaved {stem}.png/.pdf and ../../Output/n2v_refresh_networks_N100_{today}.csv")


if __name__ == "__main__":
    main()
