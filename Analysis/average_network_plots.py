"""
Network Snapshots Visualization
-------------------------------
This script creates network visualizations at multiple timesteps,
averaging across multiple simulation runs with fixed initial states.

WHAT THE NETWORKS REPRESENT:
- **Node colors**: Average opinion of each agent across all simulation runs
  (blue = cooperative/positive, red = defecting/negative)
- **Edge presence**: Edges shown only if they exist in >threshold% of runs
- **Edge width**: Thickness indicates frequency (thicker = more consistent across runs)  
- **Edge color**: Plasma colormap showing frequency (purple=low, yellow=high)
- **Edge transparency**: Lower frequency edges are more transparent

This reveals which network connections are stable vs. dynamic under different
rewiring algorithms, while showing the average opinion evolution at each position.
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
import copy
import random

# Fix import path
sys.path.append('../../')
import run
sys.path.append('..')
import models_checks

def create_fixed_initial_states(simulation_params):
    """Create fixed initial agent states that will be reused across all runs"""
    # Set fixed seed for reproducible initial configuration
    np.random.seed(42)
    random.seed(42)
    
    # Create a reference model to get network structure and initial states
    ref_model = models_checks.simulate(0, simulation_params)
    
    # Extract initial agent states
    initial_states = {}
    for node in ref_model.graph.nodes():
        initial_states[node] = ref_model.graph.nodes[node]['agent'].state
    
    return initial_states, ref_model.graph.copy()

def simulate_with_snapshots(i, simulation_params, initial_states=None):
    """Run simulation with fixed initial states if provided"""
    snapshot_timesteps = simulation_params.pop('_snapshot_timesteps', None)
    
    # Set different seed for dynamics but use fixed initial states
    np.random.seed(42 + i * 1000)  # Different seed for each run's dynamics
    random.seed(42 + i * 1000)
    
    model = models_checks.simulate(i, simulation_params)
    
    # Override with fixed initial states if provided
    if initial_states:
        for node, state in initial_states.items():
            if node in model.graph.nodes():
                model.graph.nodes[node]['agent'].state = state
        
        # Reset model tracking arrays and re-run simulation from fixed state
        model.ratio = []
        model.states = []
        model.statesds = []
        model.clustering = []
        model.degrees = []
        model.degreesSD = []
        model.mindegrees_l = []
        model.maxdegrees_l = []
        model.clusteravg = []
        model.clusterSD = []
        
        # Re-run simulation from fixed initial state
        model.runSim(simulation_params.get('timesteps', 1000), clusters=True, drawModel=False)
    
    # Capture snapshots at specified timesteps
    snapshots = {}
    if snapshot_timesteps:
        for t in snapshot_timesteps:
            if t < len(model.states):
                snapshot = {}
                if t == 0:
                    # Use actual initial states
                    for node in model.graph.nodes():
                        snapshot[node] = initial_states.get(node, model.graph.nodes[node]['agent'].state) if initial_states else model.graph.nodes[node]['agent'].state
                else:
                    # Use current agent states (actual states, not just averages)
                    for node in model.graph.nodes():
                        snapshot[node] = model.graph.nodes[node]['agent'].state
                snapshots[t] = snapshot
    
    return {'model': model, 'snapshots': snapshots}

def run_multiple_simulations(n_runs, simulation_params, snapshot_timesteps=None):
    """Run multiple simulations with fixed initial states"""
    params_copy = simulation_params.copy()
    if snapshot_timesteps:
        params_copy['_snapshot_timesteps'] = snapshot_timesteps
    
    # Create fixed initial states for all runs
    print("Creating fixed initial agent configuration...")
    initial_states, _ = create_fixed_initial_states(params_copy.copy())
    
    num_processors = run.get_optimal_process_count()
    
    # Run simulations sequentially to maintain state consistency
    # (multiprocessing with shared initial states is more complex)
    print(f"Running {n_runs} simulations with fixed initial states...")
    results = []
    for i in range(n_runs):
        if i % 5 == 0:
            print(f"  Running simulation {i+1}/{n_runs}")
        result = simulate_with_snapshots(i, params_copy.copy(), initial_states)
        results.append(result)
    
    # Extract models and organize snapshots
    models = [result['model'] for result in results]
    snapshots_by_timestep = {t: [] for t in snapshot_timesteps if t is not None}
    for result in results:
        for t, snapshot in result['snapshots'].items():
            if t in snapshots_by_timestep:
                snapshots_by_timestep[t].append(snapshot)
    
    return models, snapshots_by_timestep

def create_average_network(models, snapshots, timestep, edge_threshold=0.1, adaptive_threshold=True):
    """Create average network with improved edge filtering and weight scaling"""
    ref_model = models[0]
    num_nodes = len(ref_model.graph.nodes)
    num_models = len(models)
    network_type = getattr(models[0], 'type', 'unknown')
    
    avg_graph = nx.DiGraph() if nx.is_directed(ref_model.graph) else nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        avg_graph.add_node(i)
    
    # Calculate average opinions at this timestep
    all_opinions = {i: [] for i in range(num_nodes)}
    
    if snapshots and timestep in snapshots and len(snapshots[timestep]) > 0:
        for snapshot in snapshots[timestep]:
            for node_id, opinion in snapshot.items():
                all_opinions[node_id].append(opinion)
    else:
        for model in models:
            for i in range(num_nodes):
                all_opinions[i].append(model.graph.nodes[i]['agent'].state)
    
    # Set average opinions
    for i in range(num_nodes):
        if all_opinions[i]:
            avg_opinion = np.mean(all_opinions[i])
            avg_graph.nodes[i]['avg_opinion'] = avg_opinion
            dummy_agent = models_checks.Agent(avg_opinion, 0.5)
            avg_graph.nodes[i]['agent'] = dummy_agent
    
    # Count edge frequencies
    edge_counts = {}
    for model in models:
        for edge in model.graph.edges():
            if not nx.is_directed(model.graph) and edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    # Adaptive threshold for dense networks
    if adaptive_threshold:
        frequencies = [count / num_models for count in edge_counts.values()]
        if len(frequencies) > 0:
            freq_median = np.median(frequencies)
            freq_75 = np.percentile(frequencies, 75)
            # Use higher threshold for dense networks
            if freq_median > 0.8:  # Very dense
                edge_threshold = max(edge_threshold, freq_75)
            elif freq_median > 0.6:  # Moderately dense  
                edge_threshold = max(edge_threshold, 0.7)
    
    # Add edges with frequency-based filtering
    edge_frequencies = []
    for edge, count in edge_counts.items():
        frequency = count / num_models
        if frequency > edge_threshold:
            avg_graph.add_edge(edge[0], edge[1], weight=frequency)
            edge_frequencies.append(frequency)
    
    # Store edge statistics for visualization scaling
    if edge_frequencies:
        avg_graph.graph['edge_freq_min'] = min(edge_frequencies)
        avg_graph.graph['edge_freq_max'] = max(edge_frequencies)
        avg_graph.graph['edge_freq_range'] = max(edge_frequencies) - min(edge_frequencies)
    
    return avg_graph

def plot_average_network(avg_graph, title="Average Network", colormap='coolwarm', 
                        params=None, ax=None, show_colorbar=True, show_legend=True, layout=None):
    """Plot average network with improved edge weight visualization"""
    network_type = params.get("type") if params else None
    
    if layout is None and network_type not in ["FB", "Twitter"]:
        # Use more spread out layout for dense networks
        if avg_graph.number_of_edges() > avg_graph.number_of_nodes() * 2:
            layout = nx.spring_layout(avg_graph, k=1.0, iterations=100, seed=42)
        else:
            layout = nx.spring_layout(avg_graph, k=0.3, iterations=50, seed=42)
    
    opinions = nx.get_node_attributes(avg_graph, "avg_opinion")
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.get_cmap(colormap).reversed()
    colors = [cmap(norm(opinions[node])) for node in avg_graph.nodes]
    
    # Improved edge weight scaling
    edge_weights = []
    edge_alphas = []
    edge_colors = []
    
    if avg_graph.number_of_edges() > 0:
        weights = [avg_graph[u][v]['weight'] for u, v in avg_graph.edges()]
        min_weight = min(weights)
        max_weight = max(weights)
        weight_range = max_weight - min_weight
        
        for u, v in avg_graph.edges():
            freq = avg_graph[u][v]['weight']
            
            # Scale width: emphasize differences in frequency
            if weight_range > 0:
                # Non-linear scaling to emphasize differences
                normalized_freq = (freq - min_weight) / weight_range
                width = 0.3 + normalized_freq * 2.5  # Range: 0.3 to 2.8
            else:
                width = 1.0
                
            # Use transparency to show frequency: lower freq = more transparent
            alpha = 0.3 + (freq * 0.7)  # Range: 0.3 to 1.0
            
            # Color edges by frequency: red=low freq, blue=high freq
            edge_color = plt.cm.plasma(freq)
            
            edge_weights.append(width)
            edge_alphas.append(alpha)
            edge_colors.append(edge_color)
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    if ax is None:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        standalone = True
    else:
        standalone = False
    
    is_directed = nx.is_directed(avg_graph)
    
    # Draw nodes first
    if layout is not None:
        nx.draw_networkx_nodes(avg_graph, pos=layout, node_color=colors, 
                              edgecolors="black", node_size=190, alpha=0.9, ax=ax)
        
        # Draw edges with individual styling
        for i, (u, v) in enumerate(avg_graph.edges()):
            nx.draw_networkx_edges(avg_graph, pos=layout, edgelist=[(u, v)],
                                 width=edge_weights[i], alpha=edge_alphas[i],
                                 edge_color=[edge_colors[i]], arrows=is_directed, ax=ax)
    else:
        nx.draw_networkx_nodes(avg_graph, node_color=colors,
                              edgecolors="black", node_size=190, alpha=0.9, ax=ax)
        
        for i, (u, v) in enumerate(avg_graph.edges()):
            nx.draw_networkx_edges(avg_graph, edgelist=[(u, v)],
                                 width=edge_weights[i], alpha=edge_alphas[i], 
                                 edge_color=[edge_colors[i]], arrows=is_directed, ax=ax)
    
    if show_colorbar:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Average Opinion Value')
    
    if show_legend and avg_graph.number_of_edges() > 0:
        # Create edge frequency legend
        ax.text(1.05, 0.7, 'Edge Frequency', transform=ax.transAxes, fontsize=10, fontweight='bold')
        
        # Show frequency color scale
        freq_cmap = plt.cm.plasma
        freq_norm = Normalize(vmin=min_weight if 'min_weight' in locals() else 0, 
                             vmax=max_weight if 'max_weight' in locals() else 1)
        freq_sm = ScalarMappable(cmap=freq_cmap, norm=freq_norm)
        
        # Mini colorbar for edge frequencies
        cbar_ax = plt.gcf().add_axes([1.05, 0.4, 0.02, 0.25])
        freq_cbar = plt.colorbar(freq_sm, cax=cbar_ax)
        freq_cbar.set_label('Edge Freq', fontsize=8)
        freq_cbar.ax.tick_params(labelsize=6)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    ax.set_title(title)
    
    # Print network statistics
    if avg_graph.number_of_edges() > 0:
        print(f"Network stats: {avg_graph.number_of_nodes()} nodes, {avg_graph.number_of_edges()} edges")
        weights = [avg_graph[u][v]['weight'] for u, v in avg_graph.edges()]
        print(f"Edge frequencies: min={min(weights):.2f}, max={max(weights):.2f}, mean={np.mean(weights):.2f}")
    
    filename = None
    if standalone:
        plt.tight_layout()
        algo = params.get("rewiringAlgorithm", "") if params else ""
        mode = params.get("rewiringMode", "") if params else ""
        network_type = params.get("type", "") if params else ""
        filename = f'../../Figs/Networks/avg_network_{title}_{network_type}_{algo}_{mode}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
    
    return layout, filename

def plot_network_snapshots(models, snapshots, timesteps, edge_threshold=0.1, params=None):
    """Plot network evolution panel with improved visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if len(timesteps) == 3:
        titles = [f"Initial (t={timesteps[0]})", f"Middle (t={timesteps[1]})", f"Final (t={timesteps[2]})"]
    else:
        titles = [f"t={t}" for t in timesteps]
    
    layout = None
    
    for i, (t, ax, title) in enumerate(zip(timesteps, axes, titles)):
        avg_graph = create_average_network(models, snapshots, t, edge_threshold, adaptive_threshold=True)
        
        show_colorbar = (i == len(timesteps) - 1)
        show_legend = (i == len(timesteps) - 1)
        
        layout, _ = plot_average_network(
            avg_graph, title=title, params=params, ax=ax, 
            show_colorbar=show_colorbar, show_legend=show_legend, layout=layout
        )
    
    algo = params.get("rewiringAlgorithm", "") if params else ""
    mode = params.get("rewiringMode", "") if params else ""
    network_type = params.get("type", "") if params else ""
    suptitle = f"Network Evolution: {network_type} - {algo} - {mode} (n={len(models)} runs, fixed initial states)"
    plt.suptitle(suptitle, fontsize=16, y=1.05)
    
    filename = f'../../Figs/Networks/network_evolution_{network_type}_{algo}_{mode}_fixed.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    #plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    
    return filename

def main():
    """Main function with improved edge threshold for different network types"""
    simulation_params = {
        "rewiringAlgorithm": "biased",
        "nwsize": 100,
        "rewiringMode": "diff", 
        "type": "cl",
        "polarisingNode_f": 0.10,
        "timesteps": 25000,
        "plot": False
    }
    
    n_runs = 3
    
    # Adaptive edge threshold based on network type
    network_type = simulation_params.get("type", "")
    if network_type in ["cl", "FB"]:  # Dense/clustered networks
        edge_threshold = 0 # Only show very persistent edges
    else:  # Sparser networks
        edge_threshold = 0.50
    
    total_timesteps = simulation_params["timesteps"]
    snapshot_timesteps = [0, total_timesteps // 2, total_timesteps - 1]
    
    print(f"Running {n_runs} simulations with fixed initial states...")
    print(f"Using edge threshold: {edge_threshold} for {network_type} network")
    models, snapshots = run_multiple_simulations(n_runs, simulation_params, snapshot_timesteps)
    
    print("Plotting network evolution panel...")
    panel_filename = plot_network_snapshots(models, snapshots, snapshot_timesteps, 
                                           edge_threshold=edge_threshold, params=simulation_params)
    print(f"Panel plot saved to {panel_filename}")
    
    print("Creating average network topology (final state)...")
    avg_graph = create_average_network(models, snapshots, snapshot_timesteps[-1], edge_threshold, adaptive_threshold=True)
    
    title = f"Average Network (n={n_runs}, fixed init) - {simulation_params['type']}_{simulation_params['rewiringAlgorithm']}_{simulation_params['rewiringMode']}"
    _, filename = plot_average_network(avg_graph, title=title, params=simulation_params)
    print(f"Plot saved to {filename}")
    
    # Save data
    # avg_df, individual_df = models_checks.saveavgdata(models, "average_data.csv", simulation_params)
    # output_file = f"../../Output/avg_network_data_fixed_{simulation_params['type']}_{simulation_params['rewiringAlgorithm']}_{date.today()}.csv"
    # avg_df.to_csv(output_file, index=False)
    # print(f"Data saved to {output_file}")
    
    return models, snapshots, panel_filename

if __name__ == "__main__":
    main()