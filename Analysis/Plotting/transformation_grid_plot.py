"""
Network Transformation Grid Visualization
Creates compact grid showing network evolution across key algorithms
"""
import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import random
from datetime import date

sys.path.append('../../')
sys.path.append('..')
import models_checks

# Styling parameters
cm = 1/2.54
FONT_SIZE = 7
line_params = {
    "axis_line_width": 0.8,
    "tick_major_width": 0.8,
}

def set_plot_style():
    """Set consistent style for publication"""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'axes.linewidth': line_params["axis_line_width"],
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

def run_sims(n_runs, params, timesteps):
    """Run n simulations and return models with network snapshots

    Note: Snapshots are only saved for even-numbered runs (i % 2 == 0)
    """
    models = []

    for i in range(n_runs):
        # Use even numbers only for snapshot compatibility
        run_id = i * 2
        np.random.seed(42 + run_id * 1000)
        random.seed(42 + run_id * 1000)
        result = models_checks.simulate(run_id, params)

        # Unpack if tuple (when save_snapshots=True and i is even)
        if isinstance(result, tuple):
            model, snapshots = result
            model.snapshots = snapshots
        else:
            model = result
            model.snapshots = {}  # Initialize empty if not returned

        models.append(model)

    return models

def create_avg_network(models, t, threshold=0.3, init_graph=None):
    """Create average network with edge frequency filtering"""
    n_models = len(models)

    # Get initial topology for marking rewired edges
    if init_graph is None:
        init_graph = models[0].snapshots[0] if 0 in models[0].snapshots else models[0].graph

    init_edges = set(init_graph.edges())
    if not nx.is_directed(init_graph):
        init_edges = {tuple(sorted(e)) for e in init_edges}

    n_nodes = len(init_graph.nodes())

    # Average opinions and count edge frequencies from snapshots at time t
    avg_opinions = {i: [] for i in range(n_nodes)}
    edge_counts = {}

    for m in models:
        # Get network at timestep t
        if t in m.snapshots:
            graph_t = m.snapshots[t]
        else:
            graph_t = m.graph  # fallback to final state

        # Collect opinions
        for i in range(n_nodes):
            if i in graph_t.nodes():
                avg_opinions[i].append(graph_t.nodes[i]['agent'].state)

        # Count edges
        edges = set(graph_t.edges())
        if not nx.is_directed(graph_t):
            edges = {tuple(sorted(e)) for e in edges}
        for e in edges:
            edge_counts[e] = edge_counts.get(e, 0) + 1

    # Calculate mean opinions
    avg_opinions = {i: np.mean(opinions) if opinions else 0 for i, opinions in avg_opinions.items()}

    # Build average graph
    G = nx.DiGraph() if nx.is_directed(init_graph) else nx.Graph()
    G.add_nodes_from([(i, {'avg_opinion': avg_opinions[i]}) for i in range(n_nodes)])

    for e, count in edge_counts.items():
        freq = count / n_models
        if freq > threshold:
            is_rewired = e not in init_edges
            G.add_edge(e[0], e[1], weight=freq, rewired=is_rewired)

    return G

def plot_network_compact(G, ax, layout=None, show_cbar=False):
    """Plot network in compact style for grid"""
    # Filter to largest component
    if G.number_of_nodes() > 1:
        if nx.is_directed(G):
            comps = list(nx.weakly_connected_components(G))
        else:
            comps = list(nx.connected_components(G))

        if len(comps) > 1:
            largest = max(comps, key=len)
            G = G.subgraph(largest).copy()

    # Layout
    if layout is None:
        layout = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    else:
        # Filter layout to current nodes
        layout = {n: pos for n, pos in layout.items() if n in G.nodes()}
        # If any nodes are missing from layout, recompute it
        if len(layout) < G.number_of_nodes():
            layout = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Node colors by opinion
    opinions = [G.nodes[n]['avg_opinion'] for n in G.nodes()]
    norm = Normalize(-1, 1)
    colors = plt.cm.coolwarm_r(norm(opinions))

    # Draw nodes
    nx.draw_networkx_nodes(G, layout, node_color=colors, node_size=12,
                          edgecolors='black', linewidths=0.25, ax=ax)

    # Draw edges with rewiring distinction
    if G.number_of_edges() > 0:
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        w_min, w_max = min(weights), max(weights)
        w_range = w_max - w_min if w_max > w_min else 1

        for u, v in G.edges():
            freq = G[u][v]['weight']
            is_rewired = G[u][v].get('rewired', False)
            width = 0.1 + (freq - w_min) / w_range * 0.6
            alpha = 0.2 + freq * 0.35
            color = plt.cm.plasma(freq) if is_rewired else plt.cm.gray(0.25 + freq * 0.5)

            nx.draw_networkx_edges(G, layout, [(u, v)], width=width, alpha=alpha,
                                 edge_color=[color], arrows=False, ax=ax)

    # Calculate and display cooperation value
    avg_coop = np.mean(opinions)
    ax.text(0.02, 0.98, f'⟨a⟩={avg_coop:.2f}',
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.2))

    ax.axis('off')

    return layout

def plot_transformation_grid(n_runs=30, timesteps=[0, 14999], max_timesteps=None):
    """Create the main transformation grid figure"""
    set_plot_style()

    # Algorithm configurations
    algorithms = [
        {'name': 'Static', 'algo': 'None', 'mode': 'None'},
        {'name': 'WTF', 'algo': 'wtf', 'mode': 'None'},
        {'name': 'Bridge(opp)', 'algo': 'bridge', 'mode': 'diff'},
        {'name': 'Local(sim)', 'algo': 'biased', 'mode': 'same'},
    ]

    # Base parameters (match models_checks.py defaults)
    base_params = {
        "type": "DPAH",
        "nwsize": 150,
        "degree": 8,
        "polarisingNode_f": 0.10,
        "timesteps": max_timesteps if max_timesteps else 15000,
        "plot": False,
        "friendship": 0.5,
        "friendshipSD": 0.19,
        "skew": -0.25,
        "initSD": 0.15,
        "stubbornness": 0.6,
        "politicalClimate": 0.05,
        "newPoliticalClimate": 0.05,
        "randomness": 0.10,
        "continuous": True,
        "clustering": 0.5,
        "defectorUtility": 0.0,
        "wtf_freq": 10,
        "breaklinkprob": 1,
        "establishlinkprob": 0.5,
        "seed": 42,
        "f_all": 0.5,
        "save_snapshots": True,  # Enable network snapshots
    }

    n_rows = len(algorithms)
    n_cols = len(timesteps)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.9*cm, 14*cm))
    fig.subplots_adjust(wspace=0.05, hspace=0.15, left=0.12, right=0.95, top=0.96, bottom=0.04)

    print(f"Generating transformation grid with {n_runs} runs...")

    # Run simulations for each algorithm
    for row_idx, alg in enumerate(algorithms):
        print(f"\nProcessing {alg['name']}...")

        # Set up parameters
        params = base_params.copy()
        params['rewiringAlgorithm'] = alg['algo']
        params['rewiringMode'] = alg['mode']

        # Run simulations
        models = run_sims(n_runs, params, timesteps)

        # Store layout from t=0 for consistency
        layout = None
        init_graph = None

        # Plot each timestep
        for col_idx, t in enumerate(timesteps):
            ax = axes[row_idx, col_idx]

            # Create average network using snapshots
            G = create_avg_network(models, t, threshold=0.3, init_graph=init_graph)

            # Store initial graph for rewired edge detection
            if init_graph is None and 0 in models[0].snapshots:
                init_graph = models[0].snapshots[0]

            # Plot
            layout = plot_network_compact(G, ax, layout=layout)

            # Add column titles on top row
            if row_idx == 0:
                ax.set_title(f't={t}', fontsize=FONT_SIZE, pad=3)

        # Add row labels
        axes[row_idx, 0].text(-0.15, 0.5, alg['name'],
                              transform=axes[row_idx, 0].transAxes,
                              fontsize=FONT_SIZE, rotation=90,
                              verticalalignment='center', horizontalalignment='right')

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.97, 0.15, 0.015, 0.7])
    norm = Normalize(-1, 1)
    sm = ScalarMappable(cmap=plt.cm.coolwarm_r, norm=norm)
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Opinion', fontsize=FONT_SIZE, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=FONT_SIZE-1)

    # Save figure
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Networks", exist_ok=True)
    filename = f"../../Figs/Networks/transformation_grid_N100_DPAH_n{n_runs}_{today}"

    plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)

    print(f"\n✓ Saved: {filename}.pdf")
    print(f"✓ Saved: {filename}.png")

    plt.show()

    return fig

def main():
    """Main execution"""
    print("=" * 60)
    print("Network Transformation Grid Visualization")
    print("=" * 60)

    # Start with small test
    n_runs = int(input("Number of runs (10 for test, 90 for final): ") or "10")

    # For testing, use shorter timesteps
    use_short = input("Use short run for testing? (y/n, default=n): ").lower() == 'y'
    if use_short:
        plot_transformation_grid(n_runs=n_runs, timesteps=[0, 30000], max_timesteps=30000)
    else:
        plot_transformation_grid(n_runs=n_runs)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
