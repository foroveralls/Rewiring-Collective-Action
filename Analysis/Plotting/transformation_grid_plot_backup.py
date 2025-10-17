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
    """Create average network with edge frequency filtering

    For single run (len(models)==1), returns actual network without averaging.
    For multiple runs, averages opinions and filters edges by frequency threshold.
    """
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

    # For single run, include all edges without threshold filtering
    # For multiple runs, use frequency threshold
    effective_threshold = 0.0 if n_models == 1 else threshold

    for e, count in edge_counts.items():
        freq = count / n_models
        if freq > effective_threshold:
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

    # If graph is empty or has no nodes, return early
    if G.number_of_nodes() == 0:
        # Turn off ticks and labels but keep border frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Add thin grey border
        for spine in ax.spines.values():
            spine.set_edgecolor('#BBBBBB')
            spine.set_linewidth(0.8)
            spine.set_visible(True)
        return {}

    # Create layout - use provided layout or create new one
    # Using a consistent layout prevents artificial hub formation in averaged networks
    if layout is None:
        layout = nx.spring_layout(G, k=0.18, iterations=50, seed=42)
    else:
        # Filter layout to only include nodes in G
        layout = {n: pos for n, pos in layout.items() if n in G.nodes()}

    # Create a clean drawable graph with only nodes that have layout positions
    drawable_nodes = set(G.nodes()) & set(layout.keys())
    G_clean = G.subgraph(drawable_nodes).copy()

    # Clean layout to only include drawable nodes
    layout_clean = {n: pos for n, pos in layout.items() if n in drawable_nodes}

    # Node colors by opinion
    opinions = [G_clean.nodes[n]['avg_opinion'] for n in G_clean.nodes()]
    norm = Normalize(-1, 1)
    colors = plt.cm.coolwarm_r(norm(opinions))

    # Draw nodes
    nx.draw_networkx_nodes(G_clean, layout_clean,
                          node_color=colors, node_size=8,
                          edgecolors='black', linewidths=0.3, ax=ax)

    # Draw edges with rewiring distinction (no arrows)
    if G_clean.number_of_edges() > 0:
        edges_to_draw = list(G_clean.edges())

        if len(edges_to_draw) > 0:
            weights = [G_clean[u][v]['weight'] for u, v in edges_to_draw]
            w_min, w_max = min(weights), max(weights)
            w_range = w_max - w_min if w_max > w_min else 1

            for u, v in edges_to_draw:
                # Verify both nodes are in layout before drawing
                if u not in layout_clean or v not in layout_clean:
                    continue

                freq = G_clean[u][v]['weight']
                is_rewired = G_clean[u][v].get('rewired', False)
                width = 0.08 + (freq - w_min) / w_range * 0.35
                alpha = 0.4  # Uniform alpha for clarity
                color = '#666666' if is_rewired else 'black'  # Gray for rewired, black for original

                # Draw edges with arrows for directed graphs
                nx.draw_networkx_edges(G_clean, layout_clean, [(u, v)],
                                     width=width, alpha=alpha,
                                     edge_color=[color], arrows=True,
                                     arrowsize=4, arrowstyle='->',
                                     node_size=8,
                                     connectionstyle='arc3,rad=0.1', ax=ax)

    # Calculate and display cooperation value (print but don't show on plot)
    avg_coop = np.mean(opinions) if len(opinions) > 0 else 0.0
    print(f"  Average cooperation: {avg_coop:.2f}")

    # Turn off ticks and labels but keep border frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add visible grey border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('#BBBBBB')
        spine.set_linewidth(0.8)
        spine.set_visible(True)

    return layout_clean

def plot_transformation_grid(n_runs=30, timesteps=[0, 14999], max_timesteps=None):
    """Create the main transformation grid figure with single initial network"""
    set_plot_style()

    # Algorithm configurations
    algorithms = [
        {'name': 'Static', 'algo': 'None', 'mode': 'None'},
        {'name': 'WTF', 'algo': 'wtf', 'mode': 'None'},
        {'name': 'B-opp', 'algo': 'bridge', 'mode': 'diff'},
        {'name': 'L-sim', 'algo': 'biased', 'mode': 'same'},
    ]

    # Base parameters (match models_checks.py defaults)
    base_params = {
        "type": "DPAH",
        "nwsize": 70,
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

    # Create figure with GridSpec for custom layout
    # Left column for initial network, right column for final states
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(8.9*cm, 14*cm))
    gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[1, 1],
                  wspace=0.15, hspace=0.08, left=0.08, right=0.78, top=0.96, bottom=0.04)

    # Add title indicating averaging status
    run_type = "Single Run" if n_runs == 1 else f"Averaged over {n_runs} runs"
    fig.suptitle(run_type, fontsize=FONT_SIZE, y=0.995)

    print(f"Generating transformation grid with {n_runs} runs...")

    # First, create initial network (only once)
    print("\nGenerating initial network...")
    params = base_params.copy()
    params['rewiringAlgorithm'] = 'None'
    params['rewiringMode'] = 'None'
    models_init = run_sims(n_runs, params, [0])

    init_graph = models_init[0].snapshots[0] if 0 in models_init[0].snapshots else models_init[0].graph
    G_init = create_avg_network(models_init, 0, threshold=0.3, init_graph=init_graph)

    # Plot initial network in the middle of the left column
    ax_init = fig.add_subplot(gs[:, 0])
    init_layout = plot_network_compact(G_init, ax_init, layout=None)
    ax_init.set_title('Initial Network\n(t=0)', fontsize=FONT_SIZE, pad=5)

    # Run simulations for each algorithm and plot final states
    final_axes = []  # Store axes for arrow drawing
    for row_idx, alg in enumerate(algorithms):
        print(f"\nProcessing {alg['name']}...")

        # Set up parameters
        params = base_params.copy()
        params['rewiringAlgorithm'] = alg['algo']
        params['rewiringMode'] = alg['mode']

        # Run simulations - only need final timestep
        final_t = timesteps[-1]
        models = run_sims(n_runs, params, [0, final_t])

        # Create average network for final timestep
        G_final = create_avg_network(models, final_t, threshold=0.3, init_graph=init_graph)

        # Plot final network using the same layout as initial (prevents artificial hubs)
        ax_final = fig.add_subplot(gs[row_idx, 1])
        plot_network_compact(G_final, ax_final, layout=init_layout)

        # Add algorithm label to the right of final network
        ax_final.text(1.05, 0.5, alg['name'],
                     transform=ax_final.transAxes,
                     fontsize=FONT_SIZE, rotation=270,
                     verticalalignment='center', horizontalalignment='left')

        # Add timestep label on top row
        if row_idx == 0:
            ax_final.set_title(f't={final_t}', fontsize=FONT_SIZE, pad=3)

        final_axes.append(ax_final)

    # Add arrows from initial network to each final state
    from matplotlib.patches import FancyArrowPatch
    for row_idx, ax_final in enumerate(final_axes):
        # Get positions of axes in figure coordinates
        init_bbox = ax_init.get_position()
        final_bbox = ax_final.get_position()

        # Arrow from right middle of initial network to left middle of final network
        start_x = init_bbox.x1  # Right edge of initial network
        start_y = final_bbox.y0 + (final_bbox.y1 - final_bbox.y0) / 2  # Middle of final network vertically
        end_x = final_bbox.x0  # Left edge of final network
        end_y = start_y

        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            transform=fig.transFigure,
            arrowstyle='-|>',  # Solid filled arrowhead
            mutation_scale=8,  # Smaller arrows
            linewidth=0.8,
            color='black',
            alpha=1.0,
            zorder=0
        )
        fig.patches.append(arrow)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.82, 0.15, 0.025, 0.7])
    norm = Normalize(-1, 1)
    sm = ScalarMappable(cmap=plt.cm.coolwarm_r, norm=norm)
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Opinion', fontsize=FONT_SIZE, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=FONT_SIZE-1)

    # Save figure
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Networks", exist_ok=True)
    filename = f"../../Figs/Networks/transformation_grid_N100_DPAH_n{n_runs}_{today}"

    plt.savefig(f"{filename}.pdf", dpi=300)
    plt.savefig(f"{filename}.png", dpi=300)

    print(f"\n✓ Saved: {filename}.pdf")
    print(f"✓ Saved: {filename}.png")

    plt.show()

    return fig

def plot_transformation_circle(n_runs=30, timesteps=[0, 14999], max_timesteps=None):
    """Create transformation visualization with initial network in center, finals in circle around it"""
    set_plot_style()

    # Algorithm configurations
    algorithms = [
        {'name': 'Static', 'algo': 'None', 'mode': 'None'},
        {'name': 'WTF', 'algo': 'wtf', 'mode': 'None'},
        {'name': 'B-opp', 'algo': 'bridge', 'mode': 'diff'},
        {'name': 'L-sim', 'algo': 'biased', 'mode': 'same'},
    ]

    # Base parameters (match models_checks.py defaults)
    base_params = {
        "type": "DPAH",
        "nwsize": 100,
        "degree": 5,
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

    n_algs = len(algorithms)

    # Create figure with circular arrangement
    fig = plt.figure(figsize=(12*cm, 12*cm))

    # Calculate positions for circular arrangement
    # Center subplot for initial network
    center_ax = plt.subplot(3, 3, 5)  # Middle position in 3x3 grid

    # Positions for final networks around the center (top, right, bottom, left)
    final_positions = [2, 6, 8, 4]  # Grid positions for 4 algorithms

    # Add title indicating averaging status
    run_type = "Single Run" if n_runs == 1 else f"Averaged over {n_runs} runs"
    fig.suptitle(run_type, fontsize=FONT_SIZE, y=0.98)

    print(f"Generating circular transformation visualization with {n_runs} runs...")

    # First, create initial network (only once)
    print("\nGenerating initial network...")
    params = base_params.copy()
    params['rewiringAlgorithm'] = 'None'
    params['rewiringMode'] = 'None'
    models_init = run_sims(n_runs, params, [0])

    init_graph = models_init[0].snapshots[0] if 0 in models_init[0].snapshots else models_init[0].graph
    G_init = create_avg_network(models_init, 0, threshold=0.3, init_graph=init_graph)

    # Plot initial network in center
    init_layout = plot_network_compact(G_init, center_ax, layout=None)
    center_ax.set_title('Initial\n(t=0)', fontsize=FONT_SIZE, pad=5, weight='bold')

    # Run simulations for each algorithm and plot final states in circle
    final_axes = []
    for idx, alg in enumerate(algorithms):
        print(f"\nProcessing {alg['name']}...")

        # Set up parameters
        params = base_params.copy()
        params['rewiringAlgorithm'] = alg['algo']
        params['rewiringMode'] = alg['mode']

        # Run simulations - only need final timestep
        final_t = timesteps[-1]
        models = run_sims(n_runs, params, [0, final_t])

        # Create average network for final timestep
        G_final = create_avg_network(models, final_t, threshold=0.05, init_graph=init_graph)

        # Plot final network at corresponding position using same layout as initial
        ax_final = plt.subplot(3, 3, final_positions[idx])
        plot_network_compact(G_final, ax_final, layout=init_layout)

        # Add algorithm label
        ax_final.set_title(f'{alg["name"]}', fontsize=FONT_SIZE, pad=5)

        final_axes.append((ax_final, final_positions[idx]))

    # Add compact colorbar at top
    cbar_ax = fig.add_axes([0.35, 0.94, 0.3, 0.015])  # [left, bottom, width, height]
    norm = Normalize(-1, 1)
    sm = ScalarMappable(cmap=plt.cm.coolwarm_r, norm=norm)
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Opinion', fontsize=FONT_SIZE, labelpad=3)
    cbar.ax.tick_params(labelsize=FONT_SIZE-1)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save figure
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Networks", exist_ok=True)
    filename = f"../../Figs/Networks/transformation_circle_N100_DPAH_n{n_runs}_{today}"

    plt.savefig(f"{filename}.pdf", dpi=300)
    plt.savefig(f"{filename}.png", dpi=300)

    print(f"\n✓ Saved: {filename}.pdf")
    print(f"\n✓ Saved: {filename}.png")

    plt.show()

    return fig

def main():
    """Main execution"""
    print("=" * 60)
    print("Network Transformation Visualization")
    print("=" * 60)

    # Choose layout type
    print("\nLayout type:")
    print("  1 = Grid layout (initial on left, finals on right)")
    print("  2 = Circle layout (initial in center, finals around)")
    layout_choice = input("Enter layout type (default=1): ") or "1"

    # Allow user to specify number of runs (including single run)
    print("\nNumber of runs:")
    print("  1  = Single run (no averaging, actual network)")
    print("  10 = Quick test (averaged over 10 runs)")
    print("  30 = Standard (averaged over 30 runs)")
    print("  90 = Publication quality (averaged over 90 runs)")
    n_runs = int(input("Enter number of runs (default=10): ") or "10")

    # For testing, use shorter timesteps
    use_short = input("\nUse short run for testing? (y/n, default=n): ").lower() == 'y'

    # Call appropriate function
    if layout_choice == "2":
        if use_short:
            plot_transformation_circle(n_runs=n_runs, timesteps=[0, 30000], max_timesteps=25000)
        else:
            plot_transformation_circle(n_runs=n_runs)
    else:
        if use_short:
            plot_transformation_grid(n_runs=n_runs, timesteps=[0, 30000], max_timesteps=25000)
        else:
            plot_transformation_grid(n_runs=n_runs)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
