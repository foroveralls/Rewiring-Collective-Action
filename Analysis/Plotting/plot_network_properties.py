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
    """
    Calculate network properties for a given graph snapshot.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The network snapshot
        
    Returns:
    --------
    dict : Dictionary containing calculated properties
    """
    properties = {}
    
    try:
        # Clustering coefficient
        properties['clustering'] = nx.average_clustering(graph)
    except:
        properties['clustering'] = np.nan
    
    try:
        # Modularity using Louvain community detection
        communities = nx_community.louvain_communities(graph)
        properties['modularity'] = nx_community.modularity(graph, communities)
    except:
        properties['modularity'] = np.nan
    
    try:
        # Assortativity (degree assortativity)
        properties['assortativity'] = nx.degree_assortativity_coefficient(graph)
    except:
        properties['assortativity'] = np.nan
    
    # Placeholder for Gini coefficient - will be implemented later with NetIn
    properties['gini_degree'] = calculate_gini_degree(graph)
    
    return properties

def calculate_gini_degree(graph):
    """
    Placeholder for Gini coefficient calculation.
    This will be implemented with NetIn package later.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The network snapshot
        
    Returns:
    --------
    float : Gini coefficient of degree distribution (placeholder returns NaN)
    """
    # TODO: Implement with NetIn package
    # For now, calculate a basic Gini coefficient manually as placeholder
    try:
        degrees = [d for n, d in graph.degree()]
        if len(degrees) == 0:
            return np.nan
        
        # Sort degrees
        degrees = sorted(degrees)
        n = len(degrees)
        
        # Calculate Gini coefficient
        sum_diff = sum(abs(degrees[i] - degrees[j]) 
                      for i in range(n) for j in range(n))
        mean_degree = np.mean(degrees)
        
        if mean_degree == 0:
            return 0.0
            
        gini = sum_diff / (2 * n * n * mean_degree)
        return gini
    except:
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
    """
    Process snapshot data to calculate network properties over time.
    
    Parameters:
    -----------
    snapshot_data : dict
        Dictionary with scenario keys and snapshot data
        
    Returns:
    --------
    pd.DataFrame : DataFrame with network properties over time
    """
    all_data = []
    
    print(f"Processing {len(snapshot_data)} scenarios...")
    
    for scenario, runs in snapshot_data.items():
        print(f"\nProcessing scenario: {scenario}")
        print(f"  Number of runs: {len(runs) if isinstance(runs, list) else 1}")
        
        # Parse scenario information
        parts = scenario.split('_')
        network_type = parts[0] if parts else 'unknown'
        algorithm = parts[1] if len(parts) > 1 else 'none'
        rewiring_mode = parts[2] if len(parts) > 2 else 'none'
        
        # Handle both list of runs and single run data
        runs_to_process = runs if isinstance(runs, list) else [runs]
        
        for run_idx, run_data in enumerate(runs_to_process):
            print(f"  Processing run {run_idx}, type: {type(run_data)}")
            
            # Check different possible data structures
            if isinstance(run_data, dict):
                # Could be snapshots dict or metadata dict
                if 'snapshots' in run_data:
                    # Structure: {'snapshots': {timestep: graph, ...}, 'metadata': {...}}
                    snapshots = run_data['snapshots']
                    print(f"    Found snapshots dict with {len(snapshots)} timesteps")
                    
                    for timestep, graph in snapshots.items():
                        if hasattr(graph, 'nodes'):  # Check if it's a valid graph
                            properties = calculate_network_properties(graph)
                            
                            # Add metadata
                            properties.update({
                                'timestep': timestep,
                                'run': run_idx,
                                'scenario': scenario,
                                'network_type': network_type,
                                'algorithm': algorithm,
                                'rewiring_mode': rewiring_mode,
                                'scenario_grouped': f"{algorithm}_{rewiring_mode}"
                            })
                            
                            all_data.append(properties)
                        else:
                            print(f"    Warning: timestep {timestep} does not contain a valid graph")
                
                else:
                    # run_data might directly be snapshots dict {timestep: graph, ...}
                    print(f"    Checking if run_data is direct snapshots dict with keys: {list(run_data.keys())[:5]}...")
                    
                    # Check if values look like graphs or timesteps
                    sample_key = next(iter(run_data.keys())) if run_data else None
                    if sample_key is not None:
                        sample_value = run_data[sample_key]
                        if hasattr(sample_value, 'nodes'):  # Direct timestep: graph mapping
                            print(f"    Processing as direct timestep-graph mapping")
                            for timestep, graph in run_data.items():
                                if hasattr(graph, 'nodes'):
                                    properties = calculate_network_properties(graph)
                                    
                                    # Add metadata
                                    properties.update({
                                        'timestep': timestep,
                                        'run': run_idx,
                                        'scenario': scenario,
                                        'network_type': network_type,
                                        'algorithm': algorithm,
                                        'rewiring_mode': rewiring_mode,
                                        'scenario_grouped': f"{algorithm}_{rewiring_mode}"
                                    })
                                    
                                    all_data.append(properties)
                        else:
                            print(f"    Warning: Unknown data structure in run_data")
                            print(f"    Sample key: {sample_key}, Sample value type: {type(sample_value)}")
            else:
                print(f"    Warning: run_data is not a dict, type: {type(run_data)}")
    
    print(f"\nTotal data points processed: {len(all_data)}")
    
    if not all_data:
        print("No valid data processed from snapshots")
        return pd.DataFrame()
    
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

def plot_network_properties(df, output_file=None):
    """
    Create a grid plot of network properties over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with network properties data
    output_file : str, optional
        Output file path for saving the plot
    """
    if df.empty:
        print("No data to plot")
        return None
    
    # Properties to plot
    properties = ['clustering', 'modularity', 'assortativity', 'gini_degree']
    property_labels = {
        'clustering': 'Clustering Coefficient',
        'modularity': 'Modularity',
        'assortativity': 'Degree Assortativity',
        'gini_degree': 'Gini Coefficient (Degree)'
    }
    
    # Get unique network types and scenarios
    network_types = sorted(df['network_type'].unique())
    scenarios = sorted(df['scenario_grouped'].unique())
    
    # Create color mapping
    scenario_color_map = {}
    for scenario in scenarios:
        base_scenario = scenario.lower()
        scenario_color_map[scenario] = PLOT_COLORS.get(base_scenario, '#FE6900')
    
    # Set up the plot
    n_props = len(properties)
    n_networks = len(network_types)
    
    fig = plt.figure(figsize=(17.8*cm, 15*cm))
    gs = GridSpec(n_props, n_networks, figure=fig, 
                 top=0.85, bottom=0.15, hspace=0.3, wspace=0.25, left=0.1, right=0.95)
    
    # Plot each property for each network type
    for prop_idx, prop in enumerate(properties):
        for net_idx, network_type in enumerate(network_types):
            ax = fig.add_subplot(gs[prop_idx, net_idx])
            
            # Filter data for this network type
            net_data = df[df['network_type'] == network_type]
            
            # Plot each scenario
            for scenario in scenarios:
                scenario_data = net_data[net_data['scenario_grouped'] == scenario]
                
                if not scenario_data.empty:
                    # Group by timestep and calculate mean
                    grouped = scenario_data.groupby('timestep')[prop].agg(['mean', 'std']).reset_index()
                    
                    if not grouped.empty:
                        ax.plot(grouped['timestep'], grouped['mean'], 
                               color=scenario_color_map[scenario],
                               linewidth=line_params["data_line_width"],
                               label=scenario)
                        
                        # Add error bars if we have multiple runs
                        if 'std' in grouped.columns and not grouped['std'].isna().all():
                            ax.fill_between(grouped['timestep'], 
                                          grouped['mean'] - grouped['std'],
                                          grouped['mean'] + grouped['std'],
                                          color=scenario_color_map[scenario],
                                          alpha=0.2)
            
            # Configure axis
            is_bottom_row = prop_idx == n_props - 1
            is_left_col = net_idx == 0
            
            configure_axis_style(ax, show_ylabel=is_left_col, show_xlabel=is_bottom_row)
            
            # Set labels and title
            if prop_idx == 0:  # Top row
                ax.set_title(NETWORK_DISPLAY_NAMES.get(network_type, network_type))
            
            if is_left_col:  # Left column
                ax.set_ylabel(property_labels[prop])
            
            if is_bottom_row:  # Bottom row
                ax.set_xlabel('Timestep')
            
            # Set reasonable y-limits based on property
            if prop == 'clustering':
                ax.set_ylim(0, 1)
            elif prop == 'modularity':
                ax.set_ylim(0, 1)
            elif prop == 'assortativity':
                ax.set_ylim(-1, 1)
            elif prop == 'gini_degree':
                ax.set_ylim(0, 1)
    
    # Add legend
    legend_ax = fig.add_axes([0.15, 0.88, 0.7, 0.05])
    legend_ax.axis('off')
    
    color_elements = []
    for scenario in scenarios:
        base_scenario = scenario.lower()
        if base_scenario in PLOT_COLORS:
            # Get readable label
            if base_scenario == 'none_none':
                label = 'static'
            elif base_scenario == 'random_none':
                label = 'random'
            elif 'biased' in base_scenario:
                if 'same' in base_scenario:
                    label = 'local (similar)'
                else:
                    label = 'local (opposite)'
            elif 'bridge' in base_scenario:
                if 'same' in base_scenario:
                    label = 'bridge (similar)'
                else:
                    label = 'bridge (opposite)'
            elif base_scenario == 'wtf_none':
                label = 'wtf'
            elif base_scenario == 'node2vec_none':
                label = 'node2vec'
            else:
                label = scenario
            
            color_elements.append(Line2D([], [], color=PLOT_COLORS[base_scenario], label=label))
    
    if color_elements:
        legend_ax.legend(handles=color_elements, ncol=4, loc='center', 
                        frameon=True, bbox_to_anchor=(0.5, 0.5))
    
    plt.suptitle('Network Properties Evolution', y=0.95)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig

def main():
    """Main function to load data and create plots"""
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
    print(f"Network types: {df['network_type'].unique()}")
    print(f"Algorithms: {df['algorithm'].unique()}")
    
    # Create output filename
    today = date.today()
    output_file = f"../../Figs/Networks/network_properties_evolution_{today}.pdf"
    
    print("Creating plot...")
    fig = plot_network_properties(df, output_file=output_file)
    
    if fig:
        plt.show()
    
    return df

if __name__ == "__main__":
    df = main()