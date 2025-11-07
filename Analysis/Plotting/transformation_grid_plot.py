"""
Network Transformation Grid Visualization
Creates compact grid showing network evolution across key algorithms
"""
import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import random
import pickle
from datetime import date

sys.path.append('../../')
sys.path.append('..')
import models_checks

# Styling parameters
cm = 1/2.54
FONT_SIZE = 11  # Increased from 9 for better readability at 110% zoom
line_params = {
    "axis_line_width": 1.2,  # Increased from 1.0 for better visibility
    "tick_major_width": 1.2,  # Increased from 1.0 for better visibility
}
# Edge display threshold

e_k = 0.1

def create_opinion_colormap():
    """Create custom colormap: orange (defectors, -1) to blue (cooperators, +1)

    Uses specific RGBA colors from manuscript:
    - #daa342 (orange) for defectors (opinion = -1)
    - #3178ba (blue) for cooperators (opinion = +1)
    """
    # Convert hex colors to RGB (0-1 range)
    orange = np.array([0xda, 0xa3, 0x42]) / 255.0  # Defectors
    blue = np.array([0x31, 0x78, 0xba]) / 255.0    # Cooperators

    # Create colormap from orange (-1) through white (0) to blue (+1)
    colors = [orange, [1.0, 1.0, 1.0], blue]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('opinion_cmap', colors, N=n_bins)
    return cmap

def set_plot_style():
    """Set consistent style for publication"""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'axes.linewidth': line_params["axis_line_width"],
        'figure.dpi': 600,  # Increased from 300 for sharper output
        'savefig.dpi': 600,  # Increased from 300 for sharper output
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

            # Get initial state generator (only needed for model initialization)
            from scipy.stats import truncnorm
            def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
                return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

            friendshipWeightGenerator = get_truncated_normal(params["friendship"], params["friendshipSD"], 0, 1)
            initialStateGenerator = get_truncated_normal(params["skew"], params["initSD"], -1, 1)

            # Create model
            model = models_checks.Model(friendshipWeightGenerator=friendshipWeightGenerator,
                                       initialStateGenerator=initialStateGenerator,
                                       args=params)

            # Copy frozen topology WITH frozen agents and edge weights
            # This preserves identical initial conditions across all runs
            model.graph = deepcopy(frozen_init_graph)

            # SKIP populateModel_netin() - frozen_init_graph already has agents!
            # This ensures all runs start with identical initial opinions

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

def apply_radial_compression(layout, compression_radius_percentile=75, compression_factor=0.5):
    """Compress outer nodes toward center while preserving inner structure

    Args:
        layout: Dict of node positions
        compression_radius_percentile: Nodes beyond this percentile get compressed
        compression_factor: How much to compress outer nodes (0-1, lower = more compression)
    """
    if not layout:
        return layout

    pos_array = np.array(list(layout.values()))
    center = np.median(pos_array, axis=0)

    # Calculate distances from center
    distances = np.sqrt(np.sum((pos_array - center)**2, axis=1))
    threshold_dist = np.percentile(distances, compression_radius_percentile)

    compressed_layout = {}
    for node, pos in layout.items():
        dist = np.sqrt(np.sum((pos - center)**2))

        if dist > threshold_dist:
            # Apply smooth compression to outer nodes
            excess_dist = dist - threshold_dist
            compressed_excess = excess_dist * compression_factor
            new_dist = threshold_dist + compressed_excess

            # Scale position vector
            direction = (pos - center) / dist if dist > 0 else pos - center
            new_pos = center + direction * new_dist
            compressed_layout[node] = new_pos
        else:
            # Keep inner nodes unchanged
            compressed_layout[node] = pos

    return compressed_layout

def plot_network_compact(G, ax, layout=None, show_cbar=False, threshold=0.3, algorithm=None, mode=None):
    """Plot network in compact style for grid

    Args:
        threshold: The frequency threshold used when creating the averaged network.
                   Used for consistent edge width scaling across all graphs.
        algorithm: Optional algorithm name for algorithm-specific layout adjustments.
                   Used only for legibility, not to misrepresent network structure.
        mode: Optional mode ('same', 'diff', etc.) to further distinguish algorithm behavior.
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

        # Algorithm-specific layout parameters for legibility
        # Default parameters work well for most algorithms
        scaling_ratio = 2.0
        gravity = 1.0

        # N2V creates dense embedding-based clusters that need more node separation
        if algorithm == 'node2vec' or algorithm == 'N2V':
            scaling_ratio = 3.0  # Increased repulsion to prevent node overlap
            print(f"    Using N2V-specific layout: scaling_ratio={scaling_ratio}")

        # Random rewiring creates sparse random connections that can elongate severely
        # Need very strong gravity to compress the elongated structure
        elif algorithm == 'random':
            gravity = 4.0  # Very strong pull to center to combat elongation
            scaling_ratio = 2.5  # Slightly increased repulsion for better node separation
            print(f"    Using Random-specific layout: gravity={gravity}, scaling_ratio={scaling_ratio}")

        # Bridge-same (not bridge-opp) creates dense clusters that need more node separation
        elif algorithm == 'bridge' and mode == 'same':
            gravity = 1.0
            scaling_ratio = 4.0  # Increased repulsion to prevent node overlap
            print(f"    Using Bridge-same-specific layout: scaling_ratio={scaling_ratio}")

        # Higher iterations help sparse networks settle into good layouts
        #layout = nx.spring_layout(G, iterations=110, seed=123, scale=1)
        # layout = nx.spring_layout(G, k=k, method="energy", iterations=110, seed=123, scale=1)  # Requires NetworkX 3.5 / Python 3.12
        # layout = nx.nx_agraph.graphviz_layout(G, prog="dot", root=None)
        #layout = nx.arf_layout(G, a=1.1, pos=None, seed=123, scaling=1)  # ARF: Adaptively Restrained Force-directed
        #layout = nx.nx_agraph.graphviz_layout(G, prog="neato")
        #layout = nx.kamada_kawai_layout(G)
        #ForceAtlas2: Better for revealing community structure and polarization (built into NetworkX)
        layout = nx.forceatlas2_layout(
            G, pos=None,
            linlog=False,  # False for small networks (N=100)
            #weight="weight",
            strong_gravity=False,  # Let structure breathe naturally
            gravity=gravity,  # Moderate gravity - prevents extreme outliers while preserving structure
            scaling_ratio=scaling_ratio,  # Node repulsion strength
            seed=42
        )

        # Apply radial compression to keep outer nodes within bounds
        # Compress nodes beyond 70th percentile to 65% of their excess distance
        layout = apply_radial_compression(layout, compression_radius_percentile=70, compression_factor=0.65)
    else:
        # Filter layout to only include nodes in G
        layout = {n: pos for n, pos in layout.items() if n in G.nodes()}

    # Create a clean drawable graph with only nodes that have layout positions
    drawable_nodes = set(G.nodes()) & set(layout.keys())
    G_clean = G.subgraph(drawable_nodes).copy()

    # Clean layout to only include drawable nodes
    layout_clean = {n: pos for n, pos in layout.items() if n in drawable_nodes}

    # Calculate bounding box BEFORE filtering to determine what's in bounds
    if len(layout_clean) > 0:
        # Calculate bounding box using percentile approach to exclude extreme outliers
        pos_array = np.array([layout_clean[n] for n in layout_clean.keys()])

        # Use 95th percentile to define bounds - ignores the most extreme 5% of nodes
        x_positions = pos_array[:, 0]
        y_positions = pos_array[:, 1]

        # Calculate center from median (robust to outliers)
        x_center = np.median(x_positions)
        y_center = np.median(y_positions)

        # Calculate distances from center
        distances = np.sqrt((x_positions - x_center)**2 + (y_positions - y_center)**2)

        # Use IQR to detect and exclude extreme outliers
        q75, q25 = np.percentile(distances, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.8 * iqr  # Slightly more aggressive than standard (1.5 would be very aggressive)

        # Filter out extreme outliers, then use 92nd percentile of remaining
        non_outlier_distances = distances[distances <= outlier_threshold]
        max_dist = np.percentile(non_outlier_distances, 92) if len(non_outlier_distances) > 0 else np.percentile(distances, 88)

        # Add padding for aesthetics - balance between showing structure and preventing edge clipping
        padding = 0.20  # 20% padding to accommodate full network extent
        view_radius = max_dist * (1 + padding)

        # Define bounding box limits
        x_min, x_max = x_center - view_radius, x_center + view_radius
        y_min, y_max = y_center - view_radius, y_center + view_radius

        # Filter nodes that fall within the bounding box
        nodes_in_bounds = {n for n, pos in layout_clean.items()
                          if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max}

        # Remove nodes outside bounds
        G_clean = G_clean.subgraph(nodes_in_bounds).copy()
        layout_clean = {n: pos for n, pos in layout_clean.items() if n in nodes_in_bounds}

        print(f"    Removed {len(drawable_nodes) - len(nodes_in_bounds)} nodes outside bounding box")

    # Node colors by opinion
    # Use custom colormap: orange (defectors, -1) to blue (cooperators, +1)
    opinions = [G_clean.nodes[n]['avg_opinion'] for n in G_clean.nodes()]
    norm = Normalize(-1, 1)
    cmap = create_opinion_colormap()
    colors = cmap(norm(opinions))

    # Draw edges FIRST (so nodes appear on top)
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
                                     width=0.30, alpha=0.65,  # Reduced from 0.40 for cleaner appearance
                                     edge_color='black', arrows=True,
                                     arrowsize=10, arrowstyle='->',  # Reduced from 11 for cleaner appearance
                                     node_size=25,  # Reduced from 28 per user request
                                     connectionstyle='arc3,rad=0.1', ax=ax)

            # Draw rewired edges with frequency-scaled width (thicker than original)
            if rewired_edges:
                # Get weight range for scaling only rewired edge widths
                weights = [G_clean[u][v]['weight'] for u, v in rewired_edges]
                w_min, w_max = min(weights), max(weights)
                w_range = w_max - w_min if w_max > w_min else 1

                for u, v in rewired_edges:
                    freq = G_clean[u][v]['weight']
                    # Scale width from 0.45 to 0.85 (thicker than original to stand out)
                    width = 0.45 + (freq - w_min) / w_range * 0.40  # Increased from 0.35-0.75 range

                    nx.draw_networkx_edges(G_clean, layout_clean, [(u, v)],
                                         width=width, alpha=0.55,
                                         edge_color='#666666', arrows=True,
                                         arrowsize=13, arrowstyle='->',  # Increased from 11 for better visibility
                                         node_size=25,  # Reduced from 28 per user request
                                         connectionstyle='arc3,rad=0.1', ax=ax)

    # Draw nodes AFTER edges (so they appear on top and aren't occluded)
    nx.draw_networkx_nodes(G_clean, layout_clean,
                          node_color=colors, node_size=25,  # Reduced from 28 per user request
                          edgecolors='black', linewidths=0.5, ax=ax)  # Increased linewidth from 0.4

    # Calculate and display cooperation value (print but don't show on plot)
    avg_coop = np.mean(opinions) if len(opinions) > 0 else 0.0
    print(f"  Average cooperation: {avg_coop:.2f}")

    # Set axis limits to the bounding box we calculated earlier
    if len(layout_clean) > 0:
        # Reuse the bounding box we already calculated when filtering nodes
        pos_array = np.array([layout_clean[n] for n in layout_clean.keys()])
        x_positions = pos_array[:, 0]
        y_positions = pos_array[:, 1]

        # Calculate center and extent based on remaining nodes
        x_center = np.median(x_positions)
        y_center = np.median(y_positions)

        # Use max distance with padding
        distances = np.sqrt((x_positions - x_center)**2 + (y_positions - y_center)**2)
        max_dist = np.max(distances) if len(distances) > 0 else 1.0

        # Add small padding for aesthetics
        padding = 0.10  # 10% padding
        view_radius = max_dist * (1 + padding)

        # Set square aspect ratio for consistent appearance
        ax.set_xlim(x_center - view_radius, x_center + view_radius)
        ax.set_ylim(y_center - view_radius, y_center + view_radius)

        # Force equal aspect ratio to ensure square bounding box
        ax.set_aspect('equal', adjustable='box')

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
        {'name': 'N2V', 'algo': 'node2vec', 'mode': 'None'},  # Use 'node2vec' not 'n2v' - matches models_checks.py:301
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
    fig = plt.figure(figsize=(13*cm, 28*cm))  # Increased width from 12cm for larger elements
    gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[1, 1],
                  wspace=0.03, hspace=0.04, left=0.06, right=0.96, top=0.98, bottom=0.05)  # Reduced wspace from 0.10 to 0.03

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

    # Create a temporary model to properly populate the frozen graph
    # This ensures we use the official populateModel_netin() method with all its logic
    from scipy.stats import truncnorm
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    friendshipWeightGenerator = get_truncated_normal(base_params["friendship"], base_params["friendshipSD"], 0, 1)
    initialStateGenerator = get_truncated_normal(base_params["skew"], base_params["initSD"], -1, 1)

    # Create temp params with required keys for Model initialization
    temp_params = base_params.copy()
    temp_params['rewiringAlgorithm'] = 'None'  # No rewiring for initial state
    temp_params['rewiringMode'] = 'None'

    temp_model = models_checks.Model(friendshipWeightGenerator=friendshipWeightGenerator,
                                     initialStateGenerator=initialStateGenerator,
                                     args=temp_params)

    # Assign the generated network topology
    temp_model.graph = network_gen

    # Use the official method to populate agents with correct opinion distribution
    # This includes: proper opinion assignment, polarising nodes, edge weights
    temp_model.populateModel_netin(n, base_params["skew"])

    # FREEZE the fully populated graph as reference
    # CRITICAL: Convert to pure NetworkX Graph for consistent edge comparison
    frozen_init_graph = nx.Graph() if not nx.is_directed(temp_model.graph) else nx.DiGraph()
    frozen_init_graph.add_nodes_from(temp_model.graph.nodes(data=True))
    frozen_init_graph.add_edges_from(temp_model.graph.edges(data=True))

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
    ax_init.set_title('Initial, t=0', fontsize=FONT_SIZE, pad=5)

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
        plot_network_compact(G_final, ax_final, layout=final_layout, threshold=threshold, algorithm=alg['algo'], mode=alg['mode'])

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

    plt.savefig(f"{filename}.pdf", dpi=600, bbox_inches='tight')  # Increased DPI and added bbox_inches
    plt.savefig(f"{filename}.png", dpi=600, bbox_inches='tight')  # Increased DPI and added bbox_inches

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
        {'name': 'N2V', 'algo': 'node2vec', 'mode': 'None'},  # Use 'node2vec' not 'n2v' - matches models_checks.py:301
        {'name': 'B-opp', 'algo': 'bridge', 'mode': 'diff'},
        {'name': 'B-sim', 'algo': 'bridge', 'mode': 'same'},
        {'name': 'L-opp', 'algo': 'biased', 'mode': 'diff'},
        {'name': 'L-sim', 'algo': 'biased', 'mode': 'same'},
    ]

    # Base parameters (match models_checks.py defaults)
    base_params = {
        "type": "DPAH",
        "nwsize": 100,
        "degree": 2,
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

    # Create figure with simple 3x3 grid
    fig = plt.figure(figsize=(22*cm, 22*cm))  # Increased from 18x18 for better visibility

    # Simple 3x3 grid with initial network in center (position 5)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.15, hspace=0.15)  # Reduced spacing from 0.25

    # Center position (row=1, col=1) for initial network
    center_ax = fig.add_subplot(gs[1, 1])

    # Positions for 8 algorithms around the center (clockwise from top-left)
    # Grid positions: [row, col] in 3x3 grid
    # Layout:  0  1  2
    #          7 [C] 3
    #          6  5  4
    final_positions = [
        (0, 0),   # Top-left: Static
        (0, 1),   # Top-center: Random
        (0, 2),   # Top-right: WTF
        (1, 2),   # Middle-right: N2V
        (2, 2),   # Bottom-right: B-opp
        (2, 1),   # Bottom-center: B-sim
        (2, 0),   # Bottom-left: L-opp
        (1, 0),   # Middle-left: L-sim
    ]

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

    # Create a temporary model to properly populate the frozen graph
    # This ensures we use the official populateModel_netin() method with all its logic
    from scipy.stats import truncnorm
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    friendshipWeightGenerator = get_truncated_normal(base_params["friendship"], base_params["friendshipSD"], 0, 1)
    initialStateGenerator = get_truncated_normal(base_params["skew"], base_params["initSD"], -1, 1)

    # Create temp params with required keys for Model initialization
    temp_params = base_params.copy()
    temp_params['rewiringAlgorithm'] = 'None'  # No rewiring for initial state
    temp_params['rewiringMode'] = 'None'

    temp_model = models_checks.Model(friendshipWeightGenerator=friendshipWeightGenerator,
                                     initialStateGenerator=initialStateGenerator,
                                     args=temp_params)

    # Assign the generated network topology
    temp_model.graph = network_gen

    # Use the official method to populate agents with correct opinion distribution
    # This includes: proper opinion assignment, polarising nodes, edge weights
    temp_model.populateModel_netin(n, base_params["skew"])

    # FREEZE the fully populated graph as reference
    # CRITICAL: Convert to pure NetworkX Graph for consistent edge comparison
    frozen_init_graph = nx.Graph() if not nx.is_directed(temp_model.graph) else nx.DiGraph()
    frozen_init_graph.add_nodes_from(temp_model.graph.nodes(data=True))
    frozen_init_graph.add_edges_from(temp_model.graph.edges(data=True))

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
    center_ax.set_title('Initial (t=0)', fontsize=FONT_SIZE, pad=5, weight='bold')

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

        # Plot final network at corresponding position using GridSpec
        # For static (no rewiring), reuse initial layout since network doesn't change
        # For others, create new natural spring layout to show evolved topology
        row, col = final_positions[idx]
        ax_final = fig.add_subplot(gs[row, col])
        final_layout = init_layout if alg['algo'] == 'None' else None
        plot_network_compact(G_final, ax_final, layout=final_layout, threshold=threshold, algorithm=alg['algo'], mode=alg['mode'])

        # Add algorithm label
        ax_final.set_title(f'{alg["name"]}', fontsize=FONT_SIZE, pad=5)

        final_axes.append(ax_final)

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

    # GridSpec handles layout, no need for tight_layout

    # Save figure
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Networks", exist_ok=True)
    filename = f"../../Figs/Networks/transformation_circle_N100_DPAH_n{n_runs}_thresh{threshold}_{today}"

    plt.savefig(f"{filename}.pdf", dpi=600, bbox_inches='tight')  # Increased DPI and added bbox_inches
    plt.savefig(f"{filename}.png", dpi=600, bbox_inches='tight')  # Increased DPI and added bbox_inches

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
            plot_transformation_circle(n_runs=n_runs, threshold=threshold, timesteps=[0, 9999], max_timesteps=20000)
        else:
            plot_transformation_circle(n_runs=n_runs, threshold=threshold)
    else:
        if use_short:
            plot_transformation_grid(n_runs=n_runs, threshold=threshold, timesteps=[0, 9999], max_timesteps=20000)
        else:
            plot_transformation_grid(n_runs=n_runs, threshold=threshold)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
