"""
Network Snapshots Visualization
-------------------------------
This script creates network visualizations at multiple timesteps,
averaging across multiple simulation runs.
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
import run
import copy

# Import modules from parent directory
sys.path.append('..')
import models_checks

def simulate_with_snapshots(i, simulation_params):
    """
    Run a simulation and manually capture snapshots at specified timesteps
    
    Parameters:
    i -- Simulation index
    simulation_params -- Simulation parameters dictionary
    
    Returns:
    result_dict -- Dictionary containing the model and snapshots
    """
    # Extract snapshot timesteps if specified, otherwise use defaults
    params_copy = simulation_params.copy()
    snapshot_timesteps = params_copy.pop('_snapshot_timesteps', None)
    
    # Run the simulation normally
    model = models_checks.simulate(i, params_copy)
    
    # If no specific timesteps requested, we're done
    if not snapshot_timesteps:
        return {'model': model, 'snapshots': {}}
    
    # Capture snapshots at the specified timesteps
    snapshots = {}
    for t in snapshot_timesteps:
        # Only capture timesteps that exist in the model history
        if t < len(model.states):
            # Create a snapshot of the network state at timestep t
            snapshot = {}
            
            # For initial state (t=0), capture the initial opinions
            if t == 0:
                # Get the initial opinions for each node from the model
                for node in model.graph.nodes():
                    snapshot[node] = model.graph.nodes[node]['agent'].state
                
            # For other timesteps, we can only approximate using the global average
            # since the model doesn't track individual node states over time
            else:
                avg_state = model.states[t]
                for node in model.graph.nodes():
                    # Assign the average state to all nodes - this is an approximation
                    snapshot[node] = avg_state
            
            snapshots[t] = snapshot
    
    return {'model': model, 'snapshots': snapshots}

def run_multiple_simulations(n_runs, simulation_params, snapshot_timesteps=None):
    """
    Run multiple simulations and capture snapshots at specified timesteps
    
    Parameters:
    n_runs -- Number of simulation runs
    simulation_params -- Parameters for the simulation
    snapshot_timesteps -- List of timesteps to capture snapshots at
    
    Returns:
    models -- List of models
    snapshots -- Dictionary of snapshots by timestep
    """
    # Add snapshot timesteps to parameters if specified
    params_copy = simulation_params.copy()
    if snapshot_timesteps:
        params_copy['_snapshot_timesteps'] = snapshot_timesteps
    
    # Configure and create the process pool
    num_processors = run.get_optimal_process_count()
    pool = multiprocessing.Pool(
        processes=num_processors,
        initializer=models_checks.init_lock,
        initargs=(multiprocessing.Lock(),),
        maxtasksperchild=1
    )
    
    # Run simulations in parallel, collecting models and snapshots
    results = pool.starmap(
        simulate_with_snapshots,
        [(i, params_copy) for i in range(n_runs)]
    )
    
    pool.close()
    pool.join()
    
    # Extract models and snapshots from results
    models = [result['model'] for result in results]
    
    # Organize snapshots by timestep for easier access
    snapshots_by_timestep = {t: [] for t in snapshot_timesteps if t is not None}
    for result in results:
        for t, snapshot in result['snapshots'].items():
            if t in snapshots_by_timestep:
                snapshots_by_timestep[t].append(snapshot)
    
    return models, snapshots_by_timestep

def create_average_network(models, snapshots, timestep, edge_threshold=0.1):
    """
    Create an average network topology based on models and snapshot data
    
    Parameters:
    models -- List of model objects
    snapshots -- Dictionary of snapshots for this timestep
    timestep -- The timestep to create the network for
    edge_threshold -- Minimum frequency for an edge to be included
    
    Returns:
    avg_graph -- NetworkX graph with average node states and edge frequencies
    """
    # Get the first model as a reference for network structure
    ref_model = models[0]
    num_nodes = len(ref_model.graph.nodes)
    num_models = len(models)
    
    # Create a new graph with the same directedness as the original
    if nx.is_directed(ref_model.graph):
        avg_graph = nx.DiGraph()
    else:
        avg_graph = nx.Graph()
    
    # Add nodes to the average graph
    for i in range(num_nodes):
        avg_graph.add_node(i)
    
    # Collect opinions for this timestep across all models
    all_opinions = {i: [] for i in range(num_nodes)}
    
    # If we have snapshot data for this timestep, use it
    if snapshots and timestep in snapshots and len(snapshots[timestep]) > 0:
        for snapshot in snapshots[timestep]:
            for node_id, opinion in snapshot.items():
                all_opinions[node_id].append(opinion)
    # Otherwise use the final state from the models
    else:
        for model in models:
            # If the timestep is within range of the model's recorded states
            if 0 <= timestep < len(model.states):
                # Use the global average state as an approximation for all nodes
                avg_state = model.states[timestep]
                for i in range(num_nodes):
                    all_opinions[i].append(avg_state)
            else:
                # Use the final state from the model
                for i in range(num_nodes):
                    all_opinions[i].append(model.graph.nodes[i]['agent'].state)
    
    # Calculate average opinions for each node
    for i in range(num_nodes):
        if all_opinions[i]:  # Make sure we have opinions for this node
            avg_opinion = np.mean(all_opinions[i])
            avg_graph.nodes[i]['avg_opinion'] = avg_opinion
            
            # Create a dummy agent with the average opinion for visualization
            dummy_agent = models_checks.Agent(avg_opinion, 0.5)
            avg_graph.nodes[i]['agent'] = dummy_agent
    
    # Count edge occurrences across all models
    edge_counts = {}
    for model in models:
        for edge in model.graph.edges():
            if not nx.is_directed(model.graph) and edge[0] > edge[1]:
                # Ensure consistent edge representation for undirected graphs
                edge = (edge[1], edge[0])
            if edge in edge_counts:
                edge_counts[edge] += 1
            else:
                edge_counts[edge] = 1
    
    # Add edges to the average graph with weights representing frequency
    for edge, count in edge_counts.items():
        frequency = count / num_models
        if frequency > edge_threshold:  # Only include edges that appear with frequency > threshold
            avg_graph.add_edge(edge[0], edge[1], weight=frequency)
    
    return avg_graph

def plot_average_network(avg_graph, title="Average Network Topology", colormap='coolwarm', 
                        params=None, ax=None, show_colorbar=True, show_legend=True, layout=None):
    """
    Plot the average network with node colors representing average opinions
    and edge widths representing edge frequency
    
    Parameters:
    avg_graph -- NetworkX graph with average opinions and edge weights
    title -- Plot title
    colormap -- Matplotlib colormap name
    params -- Simulation parameters dictionary
    ax -- Matplotlib axis to plot on (if None, creates a new figure)
    show_colorbar -- Whether to show the colorbar
    show_legend -- Whether to show the edge weight legend
    layout -- Pre-calculated network layout (for consistent positioning)
    
    Returns:
    layout -- The network layout (for reuse in other plots)
    filename -- Path to saved figure (if saved)
    """
    # Get the network type (if available)
    network_type = params.get("type") if params else None
    
    # Get layout (could be spring layout or use existing layout for empirical networks)
    if layout is None:
        if network_type in ["FB", "Twitter"]:
            # For empirical networks, don't compute layout
            layout = None
        else:
            # Use a fixed seed for consistent layouts across plots
            layout = nx.spring_layout(avg_graph, k=0.3, iterations=50, seed=42)
    
    # Extract average opinions for coloring
    opinions = nx.get_node_attributes(avg_graph, "avg_opinion")
    
    # Normalize opinions for colormap
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.get_cmap(colormap).reversed()
    colors = [cmap(norm(opinions[node])) for node in avg_graph.nodes]
    
    # Get edge weights for line thickness
    edge_weights = [avg_graph[u][v]['weight'] * 3 for u, v in avg_graph.edges()]
    
    # Create colormap scalar mappable for the colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Create a new figure if no axis provided
    if ax is None:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        standalone = True
    else:
        standalone = False
    
    # Check if the graph is directed
    is_directed = nx.is_directed(avg_graph)
    
    if layout is not None:
        nx.draw(avg_graph, pos=layout, 
                node_color=colors, 
                with_labels=False, 
                edge_color='gray', 
                width=edge_weights,
                arrows=is_directed,
                edgecolors="black", 
                node_size=190, 
                font_size=10, 
                alpha=0.9, 
                ax=ax)
    else:
        nx.draw(avg_graph,
                node_color=colors, 
                with_labels=False, 
                edge_color='gray', 
                width=edge_weights,
                arrows=is_directed,
                edgecolors="black", 
                node_size=190, 
                font_size=10, 
                alpha=0.9, 
                ax=ax)
    
    # Add a colorbar if requested
    if show_colorbar:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Average Opinion Value')
    
    # Add edge frequency legend if requested
    if show_legend:
        ax.text(1.05, 0.5, 'Edge Frequency', transform=ax.transAxes, fontsize=10)
        for freq in [0.2, 0.5, 0.8]:
            ax.plot([1.05, 1.15], [0.45 - freq/4, 0.45 - freq/4], 
                    'k-', linewidth=freq*3, transform=ax.transAxes)
            ax.text(1.17, 0.45 - freq/4, f'{freq:.1f}', transform=ax.transAxes, fontsize=8, 
                    verticalalignment='center')
    
    # Add black border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    # Add title to the subplot
    ax.set_title(title)
    
    # Only handle saving and showing if this is a standalone plot
    filename = None
    if standalone:
        # Adjust layout and save
        plt.tight_layout()
        
        # Save figure - follow original naming convention
        algo = params.get("rewiringAlgorithm", "") if params else ""
        mode = params.get("rewiringMode", "") if params else ""
        network_type = params.get("type", "") if params else ""
        filename = f'../Figs/Networks/avg_network_{title}_{network_type}_{algo}_{mode}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
    
    return layout, filename

def plot_network_snapshots(models, snapshots, timesteps, edge_threshold=0.1, params=None):
    """
    Plot a panel of network snapshots at different timesteps
    
    Parameters:
    models -- List of simulation models
    snapshots -- Dictionary of snapshots organized by timestep
    timesteps -- List of timesteps to display [t_start, t_middle, t_end]
    edge_threshold -- Minimum edge frequency to include
    params -- Simulation parameters
    
    Returns:
    filename -- Path to saved figure
    """
    # Create the figure with a row of three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create appropriate titles
    if len(timesteps) == 3:
        titles = [f"Initial (t={timesteps[0]})", f"Middle (t={timesteps[1]})", f"Final (t={timesteps[2]})"]
    else:
        titles = [f"t={t}" for t in timesteps]
    
    # Create a network layout once and reuse it for all panels for consistency
    layout = None
    
    # Create and plot average networks for each timestep
    for i, (t, ax, title) in enumerate(zip(timesteps, axes, titles)):
        # Create average network for this timestep
        avg_graph = create_average_network(models, snapshots, t, edge_threshold)
        
        # Plot on the corresponding subplot
        # Only show colorbar and legend on the rightmost plot
        show_colorbar = (i == len(timesteps) - 1)
        show_legend = (i == len(timesteps) - 1)
        
        layout, _ = plot_average_network(
            avg_graph, 
            title=title, 
            params=params, 
            ax=ax, 
            show_colorbar=show_colorbar, 
            show_legend=show_legend,
            layout=layout  # Reuse layout for consistency
        )
    
    # Add an overall title
    algo = params.get("rewiringAlgorithm", "") if params else ""
    mode = params.get("rewiringMode", "") if params else ""
    network_type = params.get("type", "") if params else ""
    suptitle = f"Network Evolution: {network_type} - {algo} - {mode} (n={len(models)} runs)"
    plt.suptitle(suptitle, fontsize=16, y=1.05)
    
    # Save the figure
    filename = f'../Figs/Networks/network_evolution_{network_type}_{algo}_{mode}.png'
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    
    return filename

def main():
    """Main function to run simulations and generate visualizations"""
    
    # Example parameters similar to those in run.py
    simulation_params = {
        "rewiringAlgorithm": "biased",
        "nwsize": 500,
        "rewiringMode": "diff",
        "type": "cl",
        "polarisingNode_f": 0.10,
        "timesteps": 35000,
        "plot": False
    }
    
    # Number of simulation runs
    n_runs = 10  # For testing; can increase to 90 for the full analysis
    
    # Edge threshold - adjust to control visualization density
    edge_threshold = 0.6 # Show edges present in at least 10% of runs
    
    # Define timesteps for snapshots
    total_timesteps = simulation_params["timesteps"]
    snapshot_timesteps = [
        0,                       # Beginning
        total_timesteps // 2,    # Middle
        total_timesteps - 1      # End
    ]
    
    print(f"Running {n_runs} simulations with snapshots at {snapshot_timesteps}...")
    models, snapshots = run_multiple_simulations(n_runs, simulation_params, snapshot_timesteps)
    
    # Create a panel of network snapshots
    print("Plotting network evolution panel...")
    panel_filename = plot_network_snapshots(
        models, 
        snapshots,
        snapshot_timesteps, 
        edge_threshold=edge_threshold, 
        params=simulation_params
    )
    print(f"Panel plot saved to {panel_filename}")
    
    # Also create the full average network plot for reference
    print("Creating average network topology (final state)...")
    avg_graph = create_average_network(models, snapshots, snapshot_timesteps[-1], edge_threshold)
    
    print("Plotting average network (final state)...")
    title = f"Average Network (n={n_runs}) - {simulation_params['type']}_{simulation_params['rewiringAlgorithm']}_{simulation_params['rewiringMode']}"
    _, filename = plot_average_network(avg_graph, title=title, params=simulation_params)
    print(f"Plot saved to {filename}")
    
    # Save average data to CSV as done in run.py
    avg_df, individual_df = models_checks.saveavgdata(models, "average_data.csv", simulation_params)
    avg_df.to_csv(f"../Output/avg_network_data_{simulation_params['type']}_{simulation_params['rewiringAlgorithm']}_{date.today()}.csv", index=False)
    
    return models, snapshots, panel_filename

if __name__ == "__main__":
    main()