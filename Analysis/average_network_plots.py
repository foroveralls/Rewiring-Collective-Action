"""
Network Snapshots Visualization - Prioritizing Rewiring Activity
-----------------------------------------------------------------
Creates network visualizations focusing on edges created through rewiring
rather than initial topology, with support for empirical networks.
"""

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import multiprocessing
from datetime import date
import random

# Fix import path
sys.path.append('../../')
import run
sys.path.append('..')
import models_checks

def create_fixed_initial_states(simulation_params):
    """Create fixed initial agent states for reproducible runs"""
    np.random.seed(42)
    random.seed(42)
    
    ref_model = models_checks.simulate(0, simulation_params)
    initial_states = {node: ref_model.graph.nodes[node]['agent'].state 
                     for node in ref_model.graph.nodes()}
    
    return initial_states, ref_model.graph.copy()

def simulate_with_snapshots(i, simulation_params, initial_states=None):
    """Run simulation with fixed initial states"""
    snapshot_timesteps = simulation_params.pop('_snapshot_timesteps', None)
    
    np.random.seed(42 + i * 1000)
    random.seed(42 + i * 1000)
    
    model = models_checks.simulate(i, simulation_params)
    
    if initial_states:
        for node, state in initial_states.items():
            if node in model.graph.nodes():
                model.graph.nodes[node]['agent'].state = state
        
        # Reset and re-run from fixed state
        model.ratio = model.states = model.statesds = []
        model.clustering = model.degrees = model.degreesSD = []
        model.mindegrees_l = model.maxdegrees_l = []
        model.clusteravg = model.clusterSD = []
        
        model.runSim(simulation_params.get('timesteps', 1000), clusters=True, drawModel=False)
    
    # Capture snapshots
    snapshots = {}
    if snapshot_timesteps:
        for t in snapshot_timesteps:
            if t < len(model.states):
                if t == 0:
                    snapshot = {node: initial_states.get(node, model.graph.nodes[node]['agent'].state) 
                              for node in model.graph.nodes()} if initial_states else {}
                else:
                    snapshot = {node: model.graph.nodes[node]['agent'].state 
                              for node in model.graph.nodes()}
                snapshots[t] = snapshot
    
    return {'model': model, 'snapshots': snapshots}

def run_multiple_simulations(n_runs, simulation_params, snapshot_timesteps=None):
    """Run multiple simulations with fixed initial states"""
    params_copy = simulation_params.copy()
    if snapshot_timesteps:
        params_copy['_snapshot_timesteps'] = snapshot_timesteps
    
    print("Creating fixed initial agent configuration...")
    initial_states, _ = create_fixed_initial_states(params_copy.copy())
    
    print(f"Running {n_runs} simulations with fixed initial states...")
    results = []
    for i in range(n_runs):
        if i % 5 == 0:
            print(f"  Running simulation {i+1}/{n_runs}")
        result = simulate_with_snapshots(i, params_copy.copy(), initial_states)
        results.append(result)
    
    models = [result['model'] for result in results]
    snapshots_by_timestep = {t: [] for t in snapshot_timesteps if t is not None}
    for result in results:
        for t, snapshot in result['snapshots'].items():
            if t in snapshots_by_timestep:
                snapshots_by_timestep[t].append(snapshot)
    
    return models, snapshots_by_timestep

def create_average_network(models, snapshots, timestep, edge_threshold=0.1, prioritize_rewiring=True):
    """Create average network with simple, robust edge filtering"""
    ref_model = models[0]
    num_nodes = len(ref_model.graph.nodes)
    num_models = len(models)
    
    # Store initial topology for edge marking
    initial_edges = set(ref_model.graph.edges())
    if not nx.is_directed(ref_model.graph):
        initial_edges = {(min(e), max(e)) for e in initial_edges}
    
    avg_graph = nx.DiGraph() if nx.is_directed(ref_model.graph) else nx.Graph()
    
    # Add nodes with average opinions
    all_opinions = {i: [] for i in range(num_nodes)}
    
    if snapshots and timestep in snapshots and len(snapshots[timestep]) > 0:
        for snapshot in snapshots[timestep]:
            for node_id, opinion in snapshot.items():
                all_opinions[node_id].append(opinion)
    else:
        for model in models:
            for i in range(num_nodes):
                all_opinions[i].append(model.graph.nodes[i]['agent'].state)
    
    for i in range(num_nodes):
        if all_opinions[i]:
            avg_opinion = np.mean(all_opinions[i])
            avg_graph.add_node(i, avg_opinion=avg_opinion, 
                             agent=models_checks.Agent(avg_opinion, 0.5))
    
    # Count ALL edge frequencies
    edge_counts = {}
    for model in models:
        current_edges = set(model.graph.edges())
        if not nx.is_directed(model.graph):
            current_edges = {(min(e), max(e)) for e in current_edges}
        
        for edge in current_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    # Add edges above threshold, mark if rewired
    edge_frequencies = []
    for edge, count in edge_counts.items():
        frequency = count / num_models
        if frequency > edge_threshold:
            is_rewired = edge not in initial_edges
            avg_graph.add_edge(edge[0], edge[1], weight=frequency, rewired=is_rewired)
            edge_frequencies.append(frequency)
    
    # Store edge statistics
    if edge_frequencies:
        avg_graph.graph['edge_freq_min'] = min(edge_frequencies)
        avg_graph.graph['edge_freq_max'] = max(edge_frequencies)
        avg_graph.graph['edge_freq_range'] = max(edge_frequencies) - min(edge_frequencies)
    
    print(f"Network: {avg_graph.number_of_nodes()} nodes, {avg_graph.number_of_edges()} edges (threshold: {edge_threshold:.2f})")
    
    return avg_graph

def plot_average_network(avg_graph, title="Average Network", params=None, ax=None, 
                        show_colorbar=True, layout=None, edge_scale=1.0):
    """Simplified network plotting"""
    if layout is None and params and params.get("type") not in ["FB", "Twitter"]:
        k = 1.0 if avg_graph.number_of_edges() > avg_graph.number_of_nodes() * 2 else 0.3
        layout = nx.spring_layout(avg_graph, k=k, iterations=50, seed=42)
    
    opinions = nx.get_node_attributes(avg_graph, "avg_opinion")
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.coolwarm_r
    colors = [cmap(norm(opinions[node])) for node in avg_graph.nodes]
    
    if ax is None:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        standalone = True
    else:
        standalone = False
    
    # Draw nodes
    draw_kwargs = {'node_color': colors, 'edgecolors': "black", 'node_size': 50, 'alpha': 0.9, 'ax': ax}
    if layout: draw_kwargs['pos'] = layout
    nx.draw_networkx_nodes(avg_graph, **draw_kwargs)
    
    # Draw edges with frequency-based styling and rewiring highlighting
    if avg_graph.number_of_edges() > 0:
        weights = [avg_graph[u][v]['weight'] for u, v in avg_graph.edges()]
        w_min, w_max = min(weights), max(weights)
        w_range = w_max - w_min if w_max > w_min else 1
        
        for u, v in avg_graph.edges():
            freq = avg_graph[u][v]['weight']
            is_rewired = avg_graph[u][v].get('rewired', False)
            
            width = edge_scale * (0.3 + ((freq - w_min) / w_range) * 2.5)
            alpha = 0.4 + freq * 0.6
            
            # Different styling for rewired vs initial edges
            if is_rewired:
                edge_color = plt.cm.plasma(freq)  # Rewired edges in plasma colormap
                alpha *= 1.2  # Slightly more opaque
            else:
                edge_color = plt.cm.gray(0.3 + freq * 0.4)  # Initial edges in gray
            
            edge_kwargs = {'edgelist': [(u, v)], 'width': width, 'alpha': min(alpha, 1.0), 
                          'edge_color': [edge_color], 'arrows': nx.is_directed(avg_graph), 'ax': ax}
            if layout: edge_kwargs['pos'] = layout
            nx.draw_networkx_edges(avg_graph, **edge_kwargs)
    
    if show_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Average Opinion')
    
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    filename = None
    if standalone:
        plt.tight_layout()
        if params:
            algo, mode, ntype = params.get("rewiringAlgorithm", ""), params.get("rewiringMode", ""), params.get("type", "")
            filename = f'../../Figs/Networks/avg_network_{title}_{ntype}_{algo}_{mode}.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
    
    return layout, filename

def plot_network_snapshots(models, snapshots, timesteps, edge_threshold=0.1, params=None, edge_scale=1.0):
    """Simplified panel plotting"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [f"t={t}" for t in timesteps]
    layout = None
    
    for i, (t, ax, title) in enumerate(zip(timesteps, axes, titles)):
        avg_graph = create_average_network(models, snapshots, t, edge_threshold)
        show_cbar = (i == len(timesteps) - 1)
        layout, _ = plot_average_network(avg_graph, title=title, params=params, ax=ax, 
                                       show_colorbar=show_cbar, layout=layout, edge_scale=edge_scale)
    
    if params:
        algo, mode, ntype = params.get("rewiringAlgorithm", ""), params.get("rewiringMode", ""), params.get("type", "")
        suptitle = f"Network Evolution: {ntype} - {algo} - {mode} (n={len(models)} runs, rewiring focus)"
        plt.suptitle(suptitle, fontsize=16, y=1.05)
        
        filename = f'../../Figs/Networks/network_evolution_{ntype}_{algo}_{mode}_rewiring.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    plt.show()
    return filename

def main():
    """Main function with empirical network support"""
    # Network configuration options
    network_configs = {
        "cl": {"type": "cl", "nwsize": 100},
        "DPAH": {"type": "DPAH", "nwsize": 100}, 
        "FB": {"type": "FB", "nwsize": 786, "top_file": "FB_graph_N_786.gpickle"},
        "Twitter": {"type": "Twitter", "nwsize": 789, "top_file": "twitter_graph_N_789.gpickle"}
    }
    
    # Select network type
    print("Available networks:")
    for i, (key, config) in enumerate(network_configs.items()):
        print(f"{i}: {key} (N={config['nwsize']})")
    
    network_idx = int(input("Select network type: "))
    selected_config = list(network_configs.values())[network_idx]
    
    simulation_params = {
        "rewiringAlgorithm": "biased",
        "rewiringMode": "diff", 
        "polarisingNode_f": 0.10,
        "timesteps": 25000,
        "plot": False,
        **selected_config
    }
    
    n_runs = 1
    
    # Simple, reliable edge threshold
    edge_threshold = 0.3 if selected_config["type"] in ["cl", "FB"] else 0.2
    edge_scale = 0.5  # Adjust this to make edges thicker (>1.0) or thinner (<1.0)
    
    total_timesteps = simulation_params["timesteps"]
    snapshot_timesteps = [0, total_timesteps // 2, total_timesteps - 1]
    
    print(f"Running {n_runs} simulations on {selected_config['type']} network...")
    print(f"Edge threshold: {edge_threshold}, Edge scale: {edge_scale}")
    
    models, snapshots = run_multiple_simulations(n_runs, simulation_params, snapshot_timesteps)
    
    print("Plotting network evolution (prioritizing rewiring)...")
    panel_filename = plot_network_snapshots(models, snapshots, snapshot_timesteps, 
                                           edge_threshold=edge_threshold, params=simulation_params, edge_scale=edge_scale)
    
    print("Creating rewiring-focused average network...")
    avg_graph = create_average_network(models, snapshots, snapshot_timesteps[-1], edge_threshold)
    
    title = f"Rewiring Network (n={n_runs}) - {simulation_params['type']}_{simulation_params['rewiringAlgorithm']}_{simulation_params['rewiringMode']}"
    _, filename = plot_average_network(avg_graph, title=title, params=simulation_params, edge_scale=edge_scale)
    
    print(f"Panel: {panel_filename}")
    print(f"Single: {filename}")
    
    return models, snapshots, panel_filename

if __name__ == "__main__":
    main()