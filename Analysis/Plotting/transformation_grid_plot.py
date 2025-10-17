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
import pickle
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
# Edge display threshold

e_k = 0.1

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

def run_sims(n_runs, params, timesteps, frozen_init_graph=None):
    """Run n simulations and return models with network snapshots

    Note: Snapshots are only saved for even-numbered runs (i % 2 == 0)

    Args:
        frozen_init_graph: If provided, all runs start from this frozen initial network topology
    """
    from copy import deepcopy
    models = []

    for i in range(n_runs):
        # Use even numbers only for snapshot compatibility
        run_id = i * 2

        # Each run uses its own seed for varied dynamics
        np.random.seed(42 + run_id * 1000)
        random.seed(42 + run_id * 1000)

        # If frozen graph provided, manually create model with that topology
        if frozen_init_graph is not None:
            # Create model instance
            n = params["nwsize"]
            m = params["degree"]

            # Get initial state generator
            from scipy.stats import truncnorm
            def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
                return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

            friendshipWeightGenerator = get_truncated_normal(params["friendship"], params["friendshipSD"], 0, 1)
            initialStateGenerator = get_truncated_normal(params["skew"], params["initSD"], -1, 1)

            # Create model
            model = models_checks.Model(friendshipWeightGenerator=friendshipWeightGenerator,
                                       initialStateGenerator=initialStateGenerator,
                                       args=params)

            # Copy frozen topology
            model.graph = deepcopy(frozen_init_graph)

            # Populate with agents (with varied seeds for different initial opinions)
            model.populateModel_netin(n, params["skew"])

            # Save initial snapshot if needed
            save_snapshots = params.get('save_snapshots', False) and (i % 2 == 0)
            if save_snapshots:
                model.snapshots = {0: deepcopy(model.graph)}
            else:
                model.snapshots = {}

            # Run simulation
            steps = params["timesteps"]
            model.runSim(steps, clusters=True, drawModel=params["plot"], save_snapshots=save_snapshots)

        else:
            # Normal path: use simulate()
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

    Args:
        init_graph: FROZEN initial topology - used to identify original vs rewired edges
    """
    n_models = len(models)

    # Must have frozen initial graph as reference
    if init_graph is None:
        raise ValueError("init_graph must be provided as frozen reference topology")

    # Get edges from frozen initial topology
    init_edges = set(init_graph.edges())
    if not nx.is_directed(init_graph):
        init_edges = {tuple(sorted(e)) for e in init_edges}

    print(f"  Reference init_graph: {type(init_graph)}, directed={nx.is_directed(init_graph)}, {len(init_edges)} edges")

    n_nodes = len(init_graph.nodes())

    # Average opinions and count edge frequencies from snapshots at time t
    avg_opinions = {i: [] for i in range(n_nodes)}
    edge_counts = {}

    for idx, m in enumerate(models):
        # Debug first model only
        if idx == 0:
            print(f"    Model 0 available snapshots: {list(m.snapshots.keys())}")

        # Get network at timestep t
        if t in m.snapshots:
            graph_netin = m.snapshots[t]
            if idx == 0:
                print(f"    Using snapshot at t={t}")
        else:
            graph_netin = m.graph  # fallback to final state
            if idx == 0:
                print(f"    WARNING: t={t} not in snapshots, using final graph")

        # Convert netin object to NetworkX Graph for consistent edge representation
        graph_t = nx.Graph() if not nx.is_directed(graph_netin) else nx.DiGraph()
        graph_t.add_nodes_from(graph_netin.nodes(data=True))
        graph_t.add_edges_from(graph_netin.edges(data=True))

        # Debug first model only
        if idx == 0:
            print(f"    Model 0 snapshot: {type(graph_netin)}, converted to {type(graph_t)}, directed={nx.is_directed(graph_t)}, {graph_t.number_of_edges()} edges")

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

    n_original = 0
    n_rewired = 0
    for e, count in edge_counts.items():
        freq = count / n_models
        if freq > effective_threshold:
            is_rewired = e not in init_edges
            G.add_edge(e[0], e[1], weight=freq, rewired=is_rewired)
            if is_rewired:
                n_rewired += 1
            else:
                n_original += 1

    print(f"  Network at t={t}: {n_original} original edges, {n_rewired} rewired edges (threshold={effective_threshold:.2f})")
    print(f"    Total init_edges: {len(init_edges)}, Total edge_counts: {len(edge_counts)}")

    # Diagnostic: For initial/static, warn if rewired edges found
    if t == 0 and n_rewired > 0:
        print(f"    WARNING: Initial network (t=0) has {n_rewired} rewired edges - this should be 0!")
        # Show first few rewired edges for debugging
        rewired_samples = [e for e in edge_counts.keys() if e not in init_edges][:5]
        init_samples = list(init_edges)[:5]
        print(f"    Sample rewired edges: {rewired_samples}")
        print(f"    Sample init edges: {init_samples}")

    return G

def plot_network_compact(G, ax, layout=None, show_cbar=False, threshold=0.3):
    """Plot network in compact style for grid

    Args:
        threshold: The frequency threshold used when creating the averaged network.
                   Used for consistent edge width scaling across all graphs.
    """
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
        # Adaptive k based on network density for better sparse network layouts
        # Calculate network density: actual edges / possible edges
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) / 2 if not nx.is_directed(G) else n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0

        # k=0.18 for normal density, slightly higher for very sparse networks
        k = 0.18 if density > 0.05 else 0.22

        # Higher iterations help sparse networks settle into good layouts
        # layout = nx.spring_layout(G, k=k, iterations=110, seed=123, scale=1)
        # layout = nx.spring_layout(G, k=k, method="energy", iterations=110, seed=123, scale=1)  # Requires NetworkX 3.5 / Python 3.12
        # layout = nx.nx_agraph.graphviz_layout(G, prog="dot", root=None)
        #layout = nx.arf_layout(G, a=1.1, pos=None, seed=123, scaling=1)  # ARF: Adaptively Restrained Force-directed
        #layout = nx.nx_agraph.graphviz_layout(G, prog="fdp")
        #ForceAtlas2: Better for revealing community structure and polarization (built into NetworkX)
        layout = nx.forceatlas2_layout(
            G, pos=None,
            linlog=False,  # False for small networks (N=100)
            #weight="weight",
            strong_gravity=False,
            gravity=1.0,
            scaling_ratio=2.0,
            seed=124
        )
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

    # Draw edges with rewiring distinction
    if G_clean.number_of_edges() > 0:
        edges_to_draw = [(u, v) for u, v in G_clean.edges()
                        if u in layout_clean and v in layout_clean]

        if len(edges_to_draw) > 0:
            # Separate into original and rewired
            original_edges = [e for e in edges_to_draw if not G_clean[e[0]][e[1]].get('rewired', False)]
            rewired_edges = [e for e in edges_to_draw if G_clean[e[0]][e[1]].get('rewired', False)]
            print(f"    Plotting: {len(original_edges)} original (black), {len(rewired_edges)} rewired (grey)")

            # Draw original edges FIRST with thin fixed width
            if original_edges:
                nx.draw_networkx_edges(G_clean, layout_clean, original_edges,
                                     width=0.12, alpha=0.7,  # Thin but clearly visible black
                                     edge_color='black', arrows=True,
                                     arrowsize=3, arrowstyle='->',
                                     node_size=8,
                                     connectionstyle='arc3,rad=0.1', ax=ax)

            # Draw rewired edges with frequency-scaled width (thicker than original)
            if rewired_edges:
                # Get weight range for scaling only rewired edge widths
                weights = [G_clean[u][v]['weight'] for u, v in rewired_edges]
                w_min, w_max = min(weights), max(weights)
                w_range = w_max - w_min if w_max > w_min else 1

                for u, v in rewired_edges:
                    freq = G_clean[u][v]['weight']
                    # Scale width from 0.15 to 0.35 (thicker than original to stand out)
                    width = 0.15 + (freq - w_min) / w_range * 0.20

                    nx.draw_networkx_edges(G_clean, layout_clean, [(u, v)],
                                         width=width, alpha=0.4,
                                         edge_color='#666666', arrows=True,
                                         arrowsize=4, arrowstyle='->',
                                         node_size=8,
                                         connectionstyle='arc3,rad=0.1', ax=ax)

    # Calculate and display cooperation value (print but don't show on plot)
    avg_coop = np.mean(opinions) if len(opinions) > 0 else 0.0
    print(f"  Average cooperation: {avg_coop:.2f}")

    # Set axis limits to center the network and provide padding
    if len(layout_clean) > 0:
        # Calculate bounding box from ALL nodes to ensure nothing goes outside
        pos_array = np.array([layout_clean[n] for n in layout_clean.keys()])

        x_min, x_max = pos_array[:, 0].min(), pos_array[:, 0].max()
        y_min, y_max = pos_array[:, 1].min(), pos_array[:, 1].max()

        # Calculate ranges - use natural aspect ratio (don't force square)
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Add 7% padding to ensure nodes don't touch borders
        padding = 0.07
        x_padded = x_range * (1 + 2 * padding)
        y_padded = y_range * (1 + 2 * padding)

        # Center both dimensions using their natural ranges
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        ax.set_xlim(x_center - x_padded/2, x_center + x_padded/2)
        ax.set_ylim(y_center - y_padded/2, y_center + y_padded/2)

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

def plot_transformation_grid(n_runs=30, threshold=0.1, timesteps=[0, 14999], max_timesteps=None):
    """Create the main transformation grid figure with averaged networks

    All algorithms run with multiple simulations, averaging opinions and edges.
    """
    set_plot_style()

    # Algorithm configurations
    algorithms = [
        {'name': 'Static', 'algo': 'None', 'mode': 'None'},
        {'name': 'Random', 'algo': 'random', 'mode': 'None'},
        {'name': 'WTF', 'algo': 'wtf', 'mode': 'None'},
        {'name': 'B-opp', 'algo': 'bridge', 'mode': 'diff'},
        {'name': 'B-sim', 'algo': 'bridge', 'mode': 'same'},
        {'name': 'L-opp', 'algo': 'biased', 'mode': 'diff'},
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

    n_rows = len(algorithms)

    # Create figure with GridSpec for custom layout
    # Left column for initial network, right column for final states
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(8.9*cm, 22*cm))  # Increased height for 7 rows
    gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[1, 1],
                  wspace=0.15, hspace=0.08, left=0.08, right=0.95, top=0.98, bottom=0.06)

    print(f"Generating transformation grid with {n_runs} runs, threshold={threshold}...")

    # First, generate and FREEZE a single initial network topology
    print("\nGenerating FROZEN initial network topology (used for all runs)...")

    # Set fixed seed for reproducible network generation
    np.random.seed(42)
    random.seed(42)

    # Generate network topology directly with fixed seed (bypasses PID-dependent seeding in simulate())
    nw_type = base_params["type"]
    n = base_params["nwsize"]
    m = base_params["degree"]

    if nw_type == "DPAH":
        from netin import DPAH
        network_gen = DPAH(n, f_m=0.5, d=0.02, h_MM=base_params["f_all"], h_mm=base_params["f_all"],
                          plo_M=2.0, plo_m=2.0, seed=42)
        network_gen.generate()
    elif nw_type == "cl" or nw_type == "PATCH":
        from netin import PATCH
        network_gen = PATCH(n=n, k=m, f_m=0.5, h_MM=base_params["f_all"], h_mm=base_params["f_all"],
                           tc=base_params["clustering"], seed=42)
        network_gen.generate()
    elif nw_type == "sf":
        network_gen = nx.barabasi_albert_graph(n, m, seed=42)
    elif nw_type == "rand":
        p = 2*m/(n-1)
        network_gen = nx.erdos_renyi_graph(n, p, seed=42)
    else:
        raise ValueError(f"Unsupported network type: {nw_type}")

    # FREEZE the initial topology as reference
    # CRITICAL: Convert to pure NetworkX Graph for consistent edge comparison
    frozen_init_graph = nx.Graph() if not nx.is_directed(network_gen) else nx.DiGraph()
    frozen_init_graph.add_nodes_from(network_gen.nodes(data=True))
    frozen_init_graph.add_edges_from(network_gen.edges(data=True))

    # Populate with agents and edge weights (using same seed for reproducibility)
    from scipy.stats import truncnorm
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    friendshipWeightGenerator = get_truncated_normal(base_params["friendship"], base_params["friendshipSD"], 0, 1)
    initialStateGenerator = get_truncated_normal(base_params["skew"], base_params["initSD"], -1, 1)

    for i in frozen_init_graph.nodes():
        # Get initial state based on node class (if available from netin generators)
        if 'm' in frozen_init_graph.nodes[i]:
            node_class = frozen_init_graph.nodes[i]['m']
            state = initialStateGenerator.rvs(1)[0] if node_class == 1 else -abs(initialStateGenerator.rvs(1)[0])
        else:
            state = initialStateGenerator.rvs(1)[0]

        agent = models_checks.Agent(state, base_params["stubbornness"], base_params.get("defectorUtility", 0.0))
        frozen_init_graph.nodes[i]['agent'] = agent

    for u, v in frozen_init_graph.edges():
        weight = friendshipWeightGenerator.rvs(1)[0]
        frozen_init_graph[u][v]['weight'] = weight

    # Verify frozen topology
    print(f"  Frozen initial network: {type(frozen_init_graph)}, directed={nx.is_directed(frozen_init_graph)}")
    print(f"    {frozen_init_graph.number_of_nodes()} nodes, {frozen_init_graph.number_of_edges()} edges")
    edge_checksum = sum(hash((min(u,v), max(u,v))) for u, v in frozen_init_graph.edges()) % 1000000
    print(f"    Network checksum: {edge_checksum}")

    # Create visualization graph for initial state (all edges are "original")
    G_init = nx.DiGraph() if nx.is_directed(frozen_init_graph) else nx.Graph()
    for n in frozen_init_graph.nodes():
        opinion = frozen_init_graph.nodes[n]['agent'].state
        G_init.add_node(n, avg_opinion=opinion)
    for u, v in frozen_init_graph.edges():
        G_init.add_edge(u, v, weight=1.0, rewired=False)

    # Plot initial network in the middle of the left column
    ax_init = fig.add_subplot(gs[:, 0])
    init_layout = plot_network_compact(G_init, ax_init, layout=None, threshold=threshold)
    ax_init.set_title('Initial Network\n(t=0)', fontsize=FONT_SIZE, pad=5)

    # Run simulations for each algorithm and plot final states
    final_axes = []  # Store axes for arrow drawing
    saved_networks = {}  # Store final networks for saving

    for row_idx, alg in enumerate(algorithms):
        print(f"\nProcessing {alg['name']}...")

        # Set up parameters
        params = base_params.copy()
        params['rewiringAlgorithm'] = alg['algo']
        params['rewiringMode'] = alg['mode']

        # Run n simulations starting from the FROZEN initial network
        final_t = timesteps[-1]
        models = run_sims(n_runs, params, [0, final_t], frozen_init_graph=frozen_init_graph)

        # Create average network for final timestep, using FROZEN topology as reference
        G_final = create_avg_network(models, final_t, threshold=threshold, init_graph=frozen_init_graph)

        # Save network for later reproduction
        saved_networks[alg['name']] = G_final

        # Plot final network
        # For static (no rewiring), reuse initial layout since network doesn't change
        # For others, create new natural spring layout to show evolved topology
        ax_final = fig.add_subplot(gs[row_idx, 1])
        final_layout = init_layout if alg['algo'] == 'None' else None
        plot_network_compact(G_final, ax_final, layout=final_layout, threshold=threshold)

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

    # Add shared colorbar at bottom
    # cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.015])  # [left, bottom, width, height]
    # norm = Normalize(-1, 1)
    # sm = ScalarMappable(cmap=plt.cm.coolwarm_r, norm=norm)
    # cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # cbar.set_label('(a)', fontsize=FONT_SIZE-1, labelpad=3)
    # cbar.ax.tick_params(labelsize=FONT_SIZE-2, length=2)
    # # Rotate tick labels diagonally
    # for label in cbar.ax.get_xticklabels():
    #     label.set_rotation(45)
    #     label.set_ha('right')

    # Save figure
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Networks", exist_ok=True)
    filename = f"../../Figs/Networks/transformation_grid_N100_DPAH_n{n_runs}_thresh{threshold}_{today}"

    plt.savefig(f"{filename}.pdf", dpi=300)
    plt.savefig(f"{filename}.png", dpi=300)

    print(f"\n✓ Saved: {filename}.pdf")
    print(f"✓ Saved: {filename}.png")

    # Save network states for reproducibility
    network_data = {
        'frozen_init_graph': frozen_init_graph,
        'G_init': G_init,
        'saved_networks': saved_networks,
        'parameters': {
            'n_runs': n_runs,
            'threshold': threshold,
            'timesteps': timesteps,
            'base_params': base_params,
            'algorithms': algorithms
        }
    }
    network_filename = f"{filename}_networks.pkl"
    with open(network_filename, 'wb') as f:
        pickle.dump(network_data, f)

    print(f"✓ Saved network states: {network_filename}")

    plt.show()

    return fig

def plot_transformation_circle(n_runs=30, threshold=0.1, timesteps=[0, 14999], max_timesteps=None):
    """Create transformation visualization with initial network in center, finals in circle around it

    All algorithms run with multiple simulations, averaging opinions and edges.
    """
    set_plot_style()

    # Algorithm configurations
    algorithms = [
        {'name': 'Static', 'algo': 'None', 'mode': 'None'},
        {'name': 'Random', 'algo': 'random', 'mode': 'None'},
        {'name': 'WTF', 'algo': 'wtf', 'mode': 'None'},
        {'name': 'B-opp', 'algo': 'bridge', 'mode': 'diff'},
        {'name': 'B-sim', 'algo': 'bridge', 'mode': 'same'},
        {'name': 'L-opp', 'algo': 'biased', 'mode': 'diff'},
        {'name': 'L-sim', 'algo': 'biased', 'mode': 'same'},
    ]

    # Base parameters (match models_checks.py defaults)
    base_params = {
        "type": "DPAH",
        "nwsize": 100,
        "degree": 6,
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
    fig = plt.figure(figsize=(16*cm, 16*cm))

    # Calculate positions for circular arrangement using 4x4 grid
    # Center subplot for initial network (using middle 4 positions: 6, 7, 10, 11)
    center_ax = plt.subplot(4, 4, 6)  # Top-left of center block

    # Positions for final networks around the center (clockwise from top-left)
    # Top row: 1, 2, 3, 4; Second row: 5, (center: 6,7), 8; Third row: 9, (center: 10,11), 12; Bottom row: 13, 14, 15, 16
    final_positions = [1, 2, 4, 8, 12, 16, 13]  # Grid positions for 7 algorithms

    print(f"Generating circular transformation visualization with {n_runs} runs, threshold={threshold}...")

    # First, generate and FREEZE a single initial network topology
    print("\nGenerating FROZEN initial network topology (used for all runs)...")

    # Set fixed seed for reproducible network generation
    seed_n = 42
    np.random.seed(seed_n)
    random.seed(seed_n)

    # Generate network topology directly with fixed seed (bypasses PID-dependent seeding in simulate())
    nw_type = base_params["type"]
    n = base_params["nwsize"]
    m = base_params["degree"]

    if nw_type == "DPAH":
        from netin import DPAH
        network_gen = DPAH(n, f_m=0.5, d=0.02, h_MM=base_params["f_all"], h_mm=base_params["f_all"],
                          plo_M=2.0, plo_m=2.0, seed=seed_n)
        network_gen.generate()
    elif nw_type == "cl" or nw_type == "PATCH":
        from netin import PATCH
        network_gen = PATCH(n=n, k=m, f_m=0.5, h_MM=base_params["f_all"], h_mm=base_params["f_all"],
                           tc=base_params["clustering"], seed=seed_n)
        network_gen.generate()
    elif nw_type == "sf":
        network_gen = nx.barabasi_albert_graph(n, m, seed=seed_n)
    elif nw_type == "rand":
        p = 2*m/(n-1)
        network_gen = nx.erdos_renyi_graph(n, p, seed=seed_n)
    else:
        raise ValueError(f"Unsupported network type: {nw_type}")

    # FREEZE the initial topology as reference
    # CRITICAL: Convert to pure NetworkX Graph for consistent edge comparison
    frozen_init_graph = nx.Graph() if not nx.is_directed(network_gen) else nx.DiGraph()
    frozen_init_graph.add_nodes_from(network_gen.nodes(data=True))
    frozen_init_graph.add_edges_from(network_gen.edges(data=True))

    # Populate with agents and edge weights (using same seed for reproducibility)
    from scipy.stats import truncnorm
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    friendshipWeightGenerator = get_truncated_normal(base_params["friendship"], base_params["friendshipSD"], 0, 1)
    initialStateGenerator = get_truncated_normal(base_params["skew"], base_params["initSD"], -1, 1)

    for i in frozen_init_graph.nodes():
        # Get initial state based on node class (if available from netin generators)
        if 'm' in frozen_init_graph.nodes[i]:
            node_class = frozen_init_graph.nodes[i]['m']
            state = initialStateGenerator.rvs(1)[0] if node_class == 1 else -abs(initialStateGenerator.rvs(1)[0])
        else:
            state = initialStateGenerator.rvs(1)[0]

        agent = models_checks.Agent(state, base_params["stubbornness"], base_params.get("defectorUtility", 0.0))
        frozen_init_graph.nodes[i]['agent'] = agent

    for u, v in frozen_init_graph.edges():
        weight = friendshipWeightGenerator.rvs(1)[0]
        frozen_init_graph[u][v]['weight'] = weight

    # Verify frozen topology
    print(f"  Frozen initial network: {type(frozen_init_graph)}, directed={nx.is_directed(frozen_init_graph)}")
    print(f"    {frozen_init_graph.number_of_nodes()} nodes, {frozen_init_graph.number_of_edges()} edges")
    edge_checksum = sum(hash((min(u,v), max(u,v))) for u, v in frozen_init_graph.edges()) % 1000000
    print(f"    Network checksum: {edge_checksum}")

    # Create visualization graph for initial state (all edges are "original")
    G_init = nx.DiGraph() if nx.is_directed(frozen_init_graph) else nx.Graph()
    for n in frozen_init_graph.nodes():
        opinion = frozen_init_graph.nodes[n]['agent'].state
        G_init.add_node(n, avg_opinion=opinion)
    for u, v in frozen_init_graph.edges():
        G_init.add_edge(u, v, weight=1.0, rewired=False)

    # Plot initial network in center
    init_layout = plot_network_compact(G_init, center_ax, layout=None, threshold=threshold)
    center_ax.set_title('Initial\n(t=0)', fontsize=FONT_SIZE, pad=5, weight='bold')

    # Run simulations for each algorithm and plot final states in circle
    final_axes = []
    saved_networks = {}  # Store final networks for saving

    for idx, alg in enumerate(algorithms):
        print(f"\nProcessing {alg['name']}...")

        # Set up parameters
        params = base_params.copy()
        params['rewiringAlgorithm'] = alg['algo']
        params['rewiringMode'] = alg['mode']

        # Run n simulations starting from the FROZEN initial network
        final_t = timesteps[-1]
        models = run_sims(n_runs, params, [0, final_t], frozen_init_graph=frozen_init_graph)

        # Create average network for final timestep, using FROZEN topology as reference
        G_final = create_avg_network(models, final_t, threshold=threshold, init_graph=frozen_init_graph)

        # Save network for later reproduction
        saved_networks[alg['name']] = G_final

        # Plot final network at corresponding position
        # For static (no rewiring), reuse initial layout since network doesn't change
        # For others, create new natural spring layout to show evolved topology
        ax_final = plt.subplot(4, 4, final_positions[idx])
        final_layout = init_layout if alg['algo'] == 'None' else None
        plot_network_compact(G_final, ax_final, layout=final_layout, threshold=threshold)

        # Add algorithm label
        ax_final.set_title(f'{alg["name"]}', fontsize=FONT_SIZE, pad=5)

        final_axes.append((ax_final, final_positions[idx]))

    # Add colorbar at bottom
    # cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.015])  # [left, bottom, width, height]
    # norm = Normalize(-1, 1)
    # sm = ScalarMappable(cmap=plt.cm.coolwarm_r, norm=norm)
    # cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # cbar.set_label('(a)', fontsize=FONT_SIZE-1, labelpad=3)
    # cbar.ax.tick_params(labelsize=FONT_SIZE-2, length=2)
    # # Rotate tick labels diagonally
    # for label in cbar.ax.get_xticklabels():
    #     label.set_rotation(45)
    #     label.set_ha('right')

    plt.tight_layout()  # Remove rect constraint since no colorbar

    # Save figure
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Networks", exist_ok=True)
    filename = f"../../Figs/Networks/transformation_circle_N100_DPAH_n{n_runs}_thresh{threshold}_{today}"

    plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)

    print(f"\n✓ Saved: {filename}.pdf")
    print(f"\n✓ Saved: {filename}.png")

    # Save network states for reproducibility
    network_data = {
        'frozen_init_graph': frozen_init_graph,
        'G_init': G_init,
        'saved_networks': saved_networks,
        'parameters': {
            'n_runs': n_runs,
            'threshold': threshold,
            'timesteps': timesteps,
            'base_params': base_params,
            'algorithms': algorithms
        }
    }
    network_filename = f"{filename}_networks.pkl"
    with open(network_filename, 'wb') as f:
        pickle.dump(network_data, f)

    print(f"✓ Saved network states: {network_filename}")

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

    # Allow user to specify number of runs
    print("\nNumber of runs to average:")
    print("  1  = Single run (no averaging)")
    print("  10 = Quick test (averaged over 10 runs)")
    print("  30 = Standard (averaged over 30 runs)")
    print("  90 = Publication quality (averaged over 90 runs)")
    n_runs = int(input("Enter number of runs (default=30): ") or "30")

    # Allow user to specify edge frequency threshold
    print("\nEdge frequency threshold:")
    print("  0.0  = Include all edges")
    print("  0.1  = Include edges in >10% of runs (default)")
    print("  0.3  = Include edges in >30% of runs")
    print("  0.5  = Include edges in >50% of runs")
    threshold = float(input("Enter threshold (default=0.1): ") or "0.1")

    # For testing, use shorter timesteps
    use_short = input("\nUse short run for testing? (y/n, default=n): ").lower() == 'y'

    # Call appropriate function
    if layout_choice == "2":
        if use_short:
            plot_transformation_circle(n_runs=n_runs, threshold=threshold, timesteps=[0, 9999], max_timesteps=25000)
        else:
            plot_transformation_circle(n_runs=n_runs, threshold=threshold)
    else:
        if use_short:
            plot_transformation_grid(n_runs=n_runs, threshold=threshold, timesteps=[0, 9999], max_timesteps=25000)
        else:
            plot_transformation_grid(n_runs=n_runs, threshold=threshold)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
