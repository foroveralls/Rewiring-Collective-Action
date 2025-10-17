"""
Plot network property evolution over time using snapshot data.
Tracks clustering coefficient, modularity, assortativity, and Gini coefficient of degree distribution.
"""

import sys
import os
# Add parent directory to path for imports (needed for models_checks module)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import glob
from datetime import date
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from matplotlib.gridspec import GridSpec
import networkx as nx
from networkx.algorithms import community as nx_community

# Import existing style parameters from plots_lines.py
line_params = {
    "data_line_width": 0.8,
    "axis_line_width": 0.8,
    "grid_line_width": 0.5,
    "tick_major_width": 0.8,
    "tick_minor_width": 0.6,
    "markersize": 3
}

cm = 1/2.54
FONT_SIZE = 7

# Color scheme matching plots_lines.py
PLOT_COLORS = {
    'none_none': '#EE7733',
    'random_none': '#0077BB',
    'biased_same': '#33BBEE',
    'biased_diff': '#009988',
    'bridge_same': '#CC3311',
    'bridge_diff': '#EE3377',
    'wtf_none': '#BBBBBB',
    'node2vec_none': '#44BB99'
}

NETWORK_DISPLAY_NAMES = {
    'cl': 'CSF',
    'DPAH': 'DPAH',
    'Twitter': 'Twitter',
    'FB': 'FB'
}

def set_plot_style():
    """Set consistent style elements for all plots"""
    sns.set_style("white")
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'axes.linewidth': line_params["axis_line_width"],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (17.8*cm, 8.9*cm),
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'mathtext.default': 'regular',
        'axes.formatter.use_mathtext': True,
        'axes.axisbelow': True
    })

def calculate_network_properties(graph):
    """Calculate network properties for a given graph snapshot."""
    properties = {}
    
    try:
        # Handle netin objects - they should work directly with NetworkX functions
        # but let's add debugging
        print(f"        Processing graph type: {type(graph)}")
        
        # Try to get basic info first
        try:
            n_nodes = graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else len(graph.nodes)
            n_edges = graph.number_of_edges() if hasattr(graph, 'number_of_edges') else len(graph.edges)
            print(f"        Graph has {n_nodes} nodes, {n_edges} edges")
        except Exception as e:
            print(f"        Could not get basic graph info: {e}")
            
        # Convert to standard NetworkX if needed
        if hasattr(graph, 'generate') or 'netin.generators' in str(type(graph)):
            print(f"        Converting netin object to NetworkX")
            # Try different conversion methods
            try:
                actual_graph = nx.Graph(graph)
            except:
                # If direct conversion fails, try extracting nodes and edges manually
                actual_graph = nx.Graph()
                actual_graph.add_nodes_from(graph.nodes())
                actual_graph.add_edges_from(graph.edges())
        else:
            actual_graph = graph
            
        # Verify the conversion worked
        try:
            n_nodes_conv = actual_graph.number_of_nodes()
            n_edges_conv = actual_graph.number_of_edges()
            print(f"        Converted graph: {n_nodes_conv} nodes, {n_edges_conv} edges")
        except Exception as e:
            print(f"        Conversion verification failed: {e}")
            
    except Exception as e:
        print(f"        Graph conversion error: {e}")
        return {'clustering': np.nan, 'modularity': np.nan, 'assortativity': np.nan, 'gini_degree': np.nan}
    
    # Calculate clustering coefficient
    try:
        if actual_graph.number_of_nodes() > 0:
            properties['clustering'] = nx.average_clustering(actual_graph)
            print(f"        Clustering: {properties['clustering']:.3f}")
        else:
            properties['clustering'] = np.nan
    except Exception as e:
        print(f"        Clustering calculation failed: {e}")
        properties['clustering'] = np.nan
    
    # Calculate modularity 
    try:
        if actual_graph.number_of_nodes() > 1 and actual_graph.number_of_edges() > 0:
            communities = nx_community.louvain_communities(actual_graph)
            properties['modularity'] = nx_community.modularity(actual_graph, communities)
            print(f"        Modularity: {properties['modularity']:.3f}")
        else:
            properties['modularity'] = np.nan
    except Exception as e:
        print(f"        Modularity calculation failed: {e}")
        properties['modularity'] = np.nan
    
    # Calculate assortativity
    try:
        if actual_graph.number_of_nodes() > 1 and actual_graph.number_of_edges() > 0:
            properties['assortativity'] = nx.degree_assortativity_coefficient(actual_graph)
            print(f"        Assortativity: {properties['assortativity']:.3f}")
        else:
            properties['assortativity'] = np.nan
    except Exception as e:
        print(f"        Assortativity calculation failed: {e}")
        properties['assortativity'] = np.nan
    
    # Calculate Gini coefficient for total degree
    properties['gini_degree'] = calculate_gini_degree(actual_graph)
    print(f"        Gini (total degree): {properties['gini_degree']:.3f}")
    
    # Calculate Gini coefficient for in-degree using netin method if available
    try:
        if hasattr(graph, 'get_gini_in_ranking'):
            properties['gini_in_degree'] = graph.get_gini_in_ranking()
            print(f"        Gini (in-degree): {properties['gini_in_degree']:.3f}")
        else:
            # Fallback: calculate manually for directed graphs
            if actual_graph.is_directed():
                in_degrees = [d for n, d in actual_graph.in_degree()]
                properties['gini_in_degree'] = calculate_gini_from_degrees(in_degrees)
                print(f"        Gini (in-degree, fallback): {properties['gini_in_degree']:.3f}")
            else:
                # For undirected graphs, in-degree = total degree
                properties['gini_in_degree'] = properties['gini_degree']
    except Exception as e:
        print(f"        Gini in-degree calculation failed: {e}")
        properties['gini_in_degree'] = np.nan
    
    return properties

def calculate_gini_degree(graph):
    """Calculate Gini coefficient of degree distribution."""
    try:
        degrees = [d for n, d in graph.degree()]
        return calculate_gini_from_degrees(degrees)
    except Exception as e:
        print(f"        Gini calculation failed: {e}")
        return np.nan

def calculate_gini_from_degrees(degrees):
    """Calculate Gini coefficient from a list of degrees."""
    try:
        if len(degrees) == 0:
            return np.nan
        
        degrees = sorted(degrees)
        n = len(degrees)
        
        if n <= 1:
            return 0.0
            
        sum_diff = sum(abs(degrees[i] - degrees[j]) 
                      for i in range(n) for j in range(n))
        mean_degree = np.mean(degrees)
        
        if mean_degree == 0:
            return 0.0
            
        gini = sum_diff / (2 * n * n * mean_degree)
        return gini
    except Exception as e:
        print(f"        Gini calculation failed: {e}")
        return np.nan

def load_snapshot_data(data_dir="../../Output"):
    """
    Load snapshot data from pickle files in the output directory.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing snapshot pickle files
        
    Returns:
    --------
    dict : Dictionary with scenario keys and snapshot data
    """
    # Look for both .gz and regular .pkl files, prioritizing .gz files
    # Handle different naming patterns: *_snapshots.pkl.gz and all_snapshots*.pkl.gz
    snapshot_files = glob.glob(os.path.join(data_dir, "*_snapshots.pkl.gz"))
    snapshot_files.extend(glob.glob(os.path.join(data_dir, "all_snapshots*.pkl.gz")))
    
    if not snapshot_files:
        snapshot_files = glob.glob(os.path.join(data_dir, "*_snapshots.pkl"))
        snapshot_files.extend(glob.glob(os.path.join(data_dir, "all_snapshots*.pkl")))
    
    if not snapshot_files:
        print(f"No snapshot files found in {data_dir}")
        return {}
    
    snapshot_data = {}
    
    for file_path in snapshot_files:
        filename = os.path.basename(file_path)
        
        try:
            # Handle both .gz and regular .pkl files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Handle different file structures
            if filename.startswith('all_snapshots'):
                # all_snapshots files contain a dictionary with scenario keys
                # Structure: {scenario_key: data, ...}
                if isinstance(data, dict):
                    for scenario_key, scenario_data in data.items():
                        if scenario_key not in snapshot_data:
                            snapshot_data[scenario_key] = []
                        snapshot_data[scenario_key].append(scenario_data)
            else:
                # Individual snapshot files
                # Extract scenario info from filename (handle both .gz and regular .pkl)
                filename_clean = filename.replace('.gz', '').replace('_snapshots.pkl', '')
                parts = filename_clean.split('_')
                
                # Create scenario key from filename
                scenario_key = '_'.join(parts[:-1])  # Remove last part (usually run number)
                
                if scenario_key not in snapshot_data:
                    snapshot_data[scenario_key] = []
                
                snapshot_data[scenario_key].append(data)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return snapshot_data

def process_snapshots_to_properties(snapshot_data):
    """Process snapshot data to calculate network properties over time."""
    all_data = []
    
    print(f"Processing {len(snapshot_data)} scenarios...")
    
    for scenario, runs in snapshot_data.items():
        print(f"\nProcessing scenario: {scenario}")
        
        # Fix scenario parsing
        parts = scenario.split('_')
        if len(parts) >= 3:
            algorithm = parts[0]
            rewiring_mode = parts[1] 
            network_type = parts[2]
        else:
            algorithm = parts[0] if parts else 'unknown'
            rewiring_mode = parts[1] if len(parts) > 1 else 'none'
            network_type = parts[-1] if len(parts) > 2 else 'unknown'
        
        runs_to_process = runs if isinstance(runs, list) else [runs]
        print(f"  Number of runs: {len(runs_to_process)}")
        
        for run_idx, run_data in enumerate(runs_to_process):
            print(f"  Processing run {run_idx}")
            
            if isinstance(run_data, dict):
                if 'snapshots' in run_data:
                    snapshots = run_data['snapshots']
                else:
                    snapshots = run_data
                    
                print(f"    Found {len(snapshots)} timesteps: {sorted(snapshots.keys())}")
                
                # snapshots is {model_run: {timestep: graph_object, ...}, ...}
                graph_list = list(snapshots.items())
                for model_run, graph_dict in graph_list:
                    for timestep, graph in graph_dict.items():
                        try:
                            properties = calculate_network_properties(graph)
                            
                            properties.update({
                                'timestep': timestep,
                                'run': run_idx,
                                'model_run': model_run,
                                'scenario': scenario,
                                'network_type': network_type,
                                'algorithm': algorithm,
                                'rewiring_mode': rewiring_mode,
                                'scenario_grouped': f"{algorithm}_{rewiring_mode}"
                            })
                            
                            all_data.append(properties)
                        except:
                            continue
    return pd.DataFrame(all_data)

def configure_axis_style(ax, show_ylabel=True, show_xlabel=True):
    """Apply axis styling consistent with plots_lines.py"""
    
    # Grid configuration
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=line_params["grid_line_width"], zorder=1)
    ax.set_axisbelow(True)
    
    # Spine configuration
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(line_params["axis_line_width"])
        spine.set_zorder(100)

    # Minor tick locators
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Tick parameters
    ax.tick_params(
        axis='both', 
        which='major', 
        direction='out', 
        length=3, 
        width=line_params["tick_major_width"],
        colors='black',
        zorder=100,
        bottom=True, top=False, left=True, right=False,
        labelbottom=show_xlabel, labeltop=False, 
        labelleft=show_ylabel, labelright=False
    )
    
    ax.tick_params(
        axis='both', 
        which='minor', 
        direction='out', 
        length=1.5, 
        width=line_params["tick_minor_width"],
        colors='black',
        zorder=100,
        bottom=True, top=False, left=True, right=False,
        labelbottom=False, labeltop=False, 
        labelleft=False, labelright=False
    )

def plot_network_properties_production(df, topology_filter=None, output_file=None):
    """
    Create a production-level grid plot with simplified metrics for final publication.
    Removes assortativity and total Gini, shortens clustering coefficient label,
    and combines topological constraints.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with network properties data
    topology_filter : str, optional
        Filter data for specific topology (e.g., 'cl', 'DPAH', 'FB', 'Twitter')
    output_file : str, optional
        Output file path for saving the plot
    """
    if df.empty:
        print("No data to plot")
        return None
    
    # Filter by topology if specified
    if topology_filter:
        df = df[df['network_type'] == topology_filter]
        if df.empty:
            print(f"No data found for topology: {topology_filter}")
            return None
    
    # Properties to plot (simplified for production)
    properties = ['clustering', 'modularity', 'gini_in_degree']
    property_labels = {
        'clustering': 'Clustering',
        'modularity': 'Modularity', 
        'gini_in_degree': 'Gini (in-degree)'
    }
    
    # Get unique algorithms and create readable labels
    scenarios = sorted(df['scenario_grouped'].unique())
    
    # Create algorithm display names and ordering
    algorithm_labels = {}
    algorithm_order = []
    
    # Keep topological constraints separate (remove static)
    for scenario in scenarios:
        base_scenario = scenario.lower()
        if base_scenario == 'none_none':
            continue  # Remove static from production plots
        elif base_scenario == 'random_none':
            label = 'Random'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif 'biased' in base_scenario:
            if 'same' in base_scenario:
                label = 'Local (Similar)'
            else:
                label = 'Local (Opposite)'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif 'bridge' in base_scenario:
            if 'same' in base_scenario:
                label = 'Bridge (Similar)'
            else:
                label = 'Bridge (Opposite)'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif base_scenario == 'wtf_none':
            label = 'WTF'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif base_scenario == 'node2vec_none':
            label = 'Node2Vec'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_algorithm_order = []
    for scenario, label in algorithm_order:
        if label not in seen:
            seen.add(label)
            unique_algorithm_order.append((scenario, label))
    
    n_algorithms = len(unique_algorithm_order)
    n_properties = len(properties)
    
    if n_algorithms == 0:
        print("No algorithms found in data")
        return None
    
    # Set up the grid: metrics (rows) × algorithms (columns)
    fig = plt.figure(figsize=(4*n_algorithms*cm, 12*cm))
    gs = GridSpec(n_properties, n_algorithms, figure=fig,
                 top=0.92, bottom=0.08, hspace=0.3, wspace=0.2,
                 left=0.08, right=0.98)
    
    # Plot grid: each metric gets a row, each algorithm gets a column
    for prop_idx, prop in enumerate(properties):
        for alg_idx, (scenario, alg_label) in enumerate(unique_algorithm_order):
            ax = fig.add_subplot(gs[prop_idx, alg_idx])
            
            # Get data for this specific algorithm
            algorithm_data = df[df['scenario_grouped'] == scenario]
            
            # Get color for this algorithm
            base_scenario = scenario.lower()
            if 'biased' in base_scenario:
                color = PLOT_COLORS.get('biased_same' if 'same' in base_scenario else 'biased_diff', '#009988')
            else:
                color = PLOT_COLORS.get(base_scenario, '#FE6900')
            
            if not algorithm_data.empty:
                # Group by timestep and calculate mean/std across all simulation runs
                grouped = algorithm_data.groupby(['timestep'])[prop].agg(['mean', 'std', 'count']).reset_index()
                
                if not grouped.empty and not grouped['mean'].isna().all():
                    # Plot mean line (averaged across all simulations)
                    ax.plot(grouped['timestep'], grouped['mean'], 
                           color=color,
                           linewidth=line_params["data_line_width"])
                    
                    # Add error bars showing variation across simulation runs
                    if not grouped['std'].isna().all() and (grouped['count'] > 1).any():
                        # Use standard error for error bars
                        stderr = grouped['std'] / np.sqrt(grouped['count'])
                        ax.fill_between(grouped['timestep'], 
                                      grouped['mean'] - stderr,
                                      grouped['mean'] + stderr,
                                      color=color,
                                      alpha=0.2)
            
            # Configure axis styling
            is_bottom_row = prop_idx == n_properties - 1
            is_left_col = alg_idx == 0
            
            configure_axis_style(ax, show_ylabel=is_left_col, show_xlabel=is_bottom_row)
            
            # Set labels
            if is_left_col:
                ax.set_ylabel(property_labels[prop], fontsize=FONT_SIZE, fontweight='bold')
            
            if prop_idx == 0:  # Top row
                ax.set_title(alg_label, fontsize=FONT_SIZE, fontweight='bold')
                
            if is_bottom_row:  # Bottom row
                ax.set_xlabel('Time, t')
                # Apply scientific notation formatting like plots_lines.py
                sci_formatter = ScalarFormatter(useMathText=True)
                sci_formatter.set_scientific(True)
                sci_formatter.set_powerlimits((-2, 1))
                ax.xaxis.set_major_formatter(sci_formatter)
            
            # Set reasonable y-limits based on property
            if prop == 'clustering':
                ax.set_ylim(0, 1)
            elif prop == 'modularity':
                ax.set_ylim(0, 1)
            elif prop == 'gini_in_degree':
                ax.set_ylim(0, 1)
    
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig

def plot_network_properties_supplementary(df, topology_filter=None, output_file=None):
    """
    Create a comprehensive grid plot of network properties over time for supplementary materials.
    This is the original full version with all metrics.
    Grid layout: metrics (rows) × algorithms (columns) for easy algorithm comparison
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with network properties data
    topology_filter : str, optional
        Filter data for specific topology (e.g., 'cl', 'DPAH', 'FB', 'Twitter')
    output_file : str, optional
        Output file path for saving the plot
    """
    if df.empty:
        print("No data to plot")
        return None
    
    # Filter by topology if specified
    if topology_filter:
        df = df[df['network_type'] == topology_filter]
        if df.empty:
            print(f"No data found for topology: {topology_filter}")
            return None
    
    # Properties to plot (comprehensive for supplementary)
    properties = ['clustering', 'modularity', 'assortativity', 'gini_degree', 'gini_in_degree']
    property_labels = {
        'clustering': 'Clustering Coefficient',
        'modularity': 'Modularity', 
        'assortativity': 'Degree Assortativity',
        'gini_degree': 'Gini (total)',
        'gini_in_degree': 'Gini (in-degree)'
    }
    
    # Get unique algorithms and create readable labels
    scenarios = sorted(df['scenario_grouped'].unique())
    
    # Create algorithm display names and ordering (full version for supplementary)
    algorithm_labels = {}
    algorithm_order = []
    
    for scenario in scenarios:
        base_scenario = scenario.lower()
        if base_scenario == 'none_none':
            label = 'Static'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif base_scenario == 'random_none':
            label = 'Random'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif 'biased' in base_scenario:
            if 'same' in base_scenario:
                label = 'Local (Similar)'
            else:
                label = 'Local (Opposite)'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif 'bridge' in base_scenario:
            if 'same' in base_scenario:
                label = 'Bridge (Similar)'
            else:
                label = 'Bridge (Opposite)'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif base_scenario == 'wtf_none':
            label = 'WTF'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
        elif base_scenario == 'node2vec_none':
            label = 'Node2Vec'
            algorithm_labels[scenario] = label
            algorithm_order.append((scenario, label))
    
    # Sort algorithms for consistent ordering
    algorithm_order.sort(key=lambda x: x[1])
    
    n_algorithms = len(algorithm_order)
    n_properties = len(properties)
    
    if n_algorithms == 0:
        print("No algorithms found in data")
        return None
    
    # Set up the grid: metrics (rows) × algorithms (columns)
    fig = plt.figure(figsize=(4*n_algorithms*cm, 16*cm))
    gs = GridSpec(n_properties, n_algorithms, figure=fig,
                 top=0.92, bottom=0.08, hspace=0.3, wspace=0.2,
                 left=0.08, right=0.98)
    
    # Plot grid: each metric gets a row, each algorithm gets a column
    for prop_idx, prop in enumerate(properties):
        for alg_idx, (scenario, alg_label) in enumerate(algorithm_order):
            ax = fig.add_subplot(gs[prop_idx, alg_idx])
            
            algorithm_data = df[df['scenario_grouped'] == scenario]
            
            # Get color for this algorithm
            base_scenario = scenario.lower()
            color = PLOT_COLORS.get(base_scenario, '#FE6900')
            
            if not algorithm_data.empty:
                # Group by timestep and calculate mean/std across all simulation runs
                grouped = algorithm_data.groupby(['timestep'])[prop].agg(['mean', 'std', 'count']).reset_index()
                
                if not grouped.empty and not grouped['mean'].isna().all():
                    # Plot mean line (averaged across all 90 simulations)
                    ax.plot(grouped['timestep'], grouped['mean'], 
                           color=color,
                           linewidth=line_params["data_line_width"])
                    
                    # Add error bars showing variation across simulation runs
                    if not grouped['std'].isna().all() and (grouped['count'] > 1).any():
                        # Use standard error for error bars
                        stderr = grouped['std'] / np.sqrt(grouped['count'])
                        ax.fill_between(grouped['timestep'], 
                                      grouped['mean'] - stderr,
                                      grouped['mean'] + stderr,
                                      color=color,
                                      alpha=0.2)
            
            # Configure axis styling
            is_bottom_row = prop_idx == n_properties - 1
            is_left_col = alg_idx == 0
            
            configure_axis_style(ax, show_ylabel=is_left_col, show_xlabel=is_bottom_row)
            
            # Set labels
            if is_left_col:
                ax.set_ylabel(property_labels[prop], fontsize=FONT_SIZE, fontweight='bold')
            
            if prop_idx == 0:  # Top row
                ax.set_title(alg_label, fontsize=FONT_SIZE, fontweight='bold')
                
            if is_bottom_row:  # Bottom row
                ax.set_xlabel('Time, t')
                # Apply scientific notation formatting like plots_lines.py
                sci_formatter = ScalarFormatter(useMathText=True)
                sci_formatter.set_scientific(True)
                sci_formatter.set_powerlimits((-2, 1))
                ax.xaxis.set_major_formatter(sci_formatter)
            
            # Set reasonable y-limits based on property
            if prop == 'clustering':
                ax.set_ylim(0, 1)
            elif prop == 'modularity':
                ax.set_ylim(0, 1)
            elif prop == 'assortativity':
                ax.set_ylim(-1, 1)
            elif prop == 'gini_degree':
                ax.set_ylim(0, 1)
            elif prop == 'gini_in_degree':
                ax.set_ylim(0, 1)
    
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig

# Keep original function name for backward compatibility
def plot_network_properties(df, topology_filter=None, output_file=None):
    """
    Wrapper function that calls the supplementary version for backward compatibility.
    """
    return plot_network_properties_supplementary(df, topology_filter, output_file)

def main(topology_filter=None, production_plots=True, supplementary_plots=False):
    """Main function to load data and create plots
    
    Parameters:
    -----------
    topology_filter : str, optional
        Specific topology to plot (e.g., 'cl', 'DPAH', 'FB', 'Twitter').
        If None, creates plots for all available topologies.
    production_plots : bool, default True
        Whether to create production-level plots with simplified metrics
    supplementary_plots : bool, default False
        Whether to create comprehensive supplementary plots with all metrics
    """
    set_plot_style()
    
    print("Loading snapshot data...")
    snapshot_data = load_snapshot_data()
    
    if not snapshot_data:
        print("No snapshot data found. Make sure you have run simulations with save_snapshots=True")
        return
    
    print(f"Found {len(snapshot_data)} scenarios with snapshot data")
    
    print("Processing snapshots to calculate network properties...")
    df = process_snapshots_to_properties(snapshot_data)
    
    if df.empty:
        print("No valid network properties data generated")
        return
    
    print(f"Processed {len(df)} data points")
    
    # Determine which topologies to plot
    if topology_filter:
        topologies_to_plot = [topology_filter]
    else:
        topologies_to_plot = df['network_type'].unique()
    
    # Create plots for each topology
    today = date.today()
    for topology in topologies_to_plot:
        # Create production-level plots if requested
        if production_plots:
            output_file = f"../../Figs/Networks/network_properties_production_{topology}_{today}.pdf"
            fig = plot_network_properties_production(df, topology_filter=topology, output_file=output_file)
            print(f"Created production plot for {topology}")
        
        # Create supplementary plots if requested  
        if supplementary_plots:
            output_file = f"../../Figs/Networks/network_properties_supplementary_{topology}_{today}.pdf"
            fig = plot_network_properties_supplementary(df, topology_filter=topology, output_file=output_file)
            print(f"Created supplementary plot for {topology}")
    
    return df

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    topology_filter = None
    production_plots = True
    supplementary_plots = False
    
    for arg in sys.argv[1:]:
        if arg in ['cl', 'DPAH', 'FB', 'Twitter']:
            topology_filter = arg
        elif arg == '--supplementary':
            supplementary_plots = True
        elif arg == '--production-only':
            production_plots = True
            supplementary_plots = False
        elif arg == '--both':
            production_plots = True
            supplementary_plots = True
        elif arg == '--help':
            print("Usage: python plot_network_properties.py [topology] [options]")
            print("  topology: cl, DPAH, FB, or Twitter (optional)")
            print("  --supplementary: also create supplementary plots with all metrics")
            print("  --production-only: create only production plots (default)")
            print("  --both: create both production and supplementary plots")
            print("  --help: show this help message")
            sys.exit(0)
    
    df = main(topology_filter=topology_filter, 
              production_plots=production_plots, 
              supplementary_plots=supplementary_plots)