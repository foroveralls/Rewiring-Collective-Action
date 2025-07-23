#!/usr/bin/env python3
"""
Custom run script for wtf and node2vec algorithms with sequential algorithm execution
"""

import os
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import date
import multiprocessing
from itertools import repeat
import models_checks

def init(lock_):
    models_checks.init_lock(lock_)

def run_algorithm(algo, mode, topology, n_sims=20):
    """Run single algorithm with multiprocessing"""
    print(f"Running {algo} on {topology}...")

    # Network config
    if topology == "Twitter":
        top_file, nwsize = "twitter_graph_N_789.gpickle", 789
    else:  # DPAH
        top_file, nwsize = None, 300

    args = {
        "rewiringAlgorithm": algo, "rewiringMode": mode, "type": topology,
        "top_file": top_file, "nwsize": nwsize, "timesteps": 60000,
        "polarisingNode_f": 0.10, "plot": False, "seed": 42
    }

    base_args = models_checks.getargs()
    complete_args = {**base_args, **args}

    # Multiprocessing within algorithm
    n_proc = max(1, int(0.7 * multiprocessing.cpu_count()))
    lock = multiprocessing.Lock()

    with multiprocessing.Pool(processes=n_proc, initializer=init, initargs=(lock,)) as pool:
        models = pool.starmap(models_checks.simulate,
                             zip(range(n_sims), repeat(complete_args)))
        pool.close()
        pool.join()

    return models

def extract_trajectories(models, algo, topology):
    """Extract trajectory data"""
    data = []
    for i, m in enumerate(models):
        for t, state in enumerate(m.states):
            data.append({
                't': t, 'avg_state': state, 'std_states': m.statesds[t],
                'model_run': i, 'algo': algo, 'topology': topology
            })
    return pd.DataFrame(data)

def plot_trajectories(wtf_data, node2vec_data):
    """Simple trajectory plot for directed networks only"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'wtf': '#BBBBBB', 'node2vec': '#44BB99'}
    topologies = ['DPAH', 'Twitter']

    for i, topo in enumerate(topologies):
        ax = axes[i]

        # Plot wtf
        if not wtf_data.empty:
            wtf_topo = wtf_data[wtf_data['topology'] == topo]
            if not wtf_topo.empty:
                wtf_avg = wtf_topo.groupby('t')['avg_state'].mean()
                ax.plot(wtf_avg.index, wtf_avg.values,
                       color=colors['wtf'], label='WTF', linewidth=1.5)

        # Plot node2vec
        if not node2vec_data.empty:
            n2v_topo = node2vec_data[node2vec_data['topology'] == topo]
            if not n2v_topo.empty:
                n2v_avg = n2v_topo.groupby('t')['avg_state'].mean()
                ax.plot(n2v_avg.index, n2v_avg.values,
                       color=colors['node2vec'], label='Node2Vec', linewidth=1.5)

        ax.set_title(topo)
        ax.set_ylim(-0.6, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time')
        if i == 0:
            ax.set_ylabel('Cooperativity')
            ax.legend()

    plt.tight_layout()
    return fig

def main():
    start = time.time()
    n_sims = 80  # Reasonable for multiprocessing
    topologies = ['DPAH', 'Twitter']

    print("=== WTF & Node2Vec Sequential Algorithm Run ===")
    print(f"Simulations per scenario: {n_sims}")
    print(f"Topologies: {topologies}")

    # Run WTF first - complete all before starting node2vec
    print("\n=== Phase 1: WTF Algorithm ===")
    wtf_models = {}
    for topo in topologies:
        wtf_models[topo] = run_algorithm('wtf', 'None', topo, n_sims)

    print("\n=== Phase 2: Node2Vec Algorithm ===")
    # Run Node2Vec second - after WTF is completely finished
    node2vec_models = {}
    for topo in topologies:
        node2vec_models[topo] = run_algorithm('node2vec', 'None', topo, n_sims)

    print("\n=== Processing Data ===")

    # Extract trajectories
    wtf_data = pd.concat([
        extract_trajectories(models, 'wtf', topo)
        for topo, models in wtf_models.items()
    ], ignore_index=True)

    node2vec_data = pd.concat([
        extract_trajectories(models, 'node2vec', topo)
        for topo, models in node2vec_models.items()
    ], ignore_index=True)

    # Save data
    today = date.today()
    wtf_data.to_csv(f'../Output/wtf_trajectories_{today}.csv', index=False)
    node2vec_data.to_csv(f'../Output/node2vec_trajectories_{today}.csv', index=False)

    # Plot
    print("=== Creating Plot ===")
    fig = plot_trajectories(wtf_data, node2vec_data)

    os.makedirs('../Figs/Custom', exist_ok=True)
    fig.savefig(f'../Figs/Custom/wtf_node2vec_comparison_{today}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

    # Summary
    elapsed = (time.time() - start) / 60
    total_sims = len(topologies) * n_sims * 2  # 2 algorithms

    print(f"\n=== Complete ===")
    print(f"Total simulations: {total_sims}")
    print(f"Runtime: {elapsed:.1f} minutes")
    print(f"Data saved: wtf_trajectories_{today}.csv, node2vec_trajectories_{today}.csv")
    print(f"Plot saved: wtf_node2vec_comparison_{today}.pdf")

if __name__ == "__main__":
    main()
