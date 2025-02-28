#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized heatmap generator for parameter sweeps.
Creates one heatmap per topology with all rewiring algorithms.
Focused on correctly processing data and generating filled heatmaps.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
import gc
import time
import multiprocessing
from functools import partial

# Constants for paper style
FONT_SIZE = 14
FRIENDLY_COLORS = {
    'static': '#EE7733',      # Orange
    'random': '#0077BB',      # Blue
    'local (similar)': '#33BBEE',    # Cyan
    'local (opposite)': '#009988',   # Teal
    'bridge (similar)': '#CC3311',   # Red
    'bridge (opposite)': '#EE3377',  # Magenta
    'wtf': '#BBBBBB',         # Grey
    'node2vec': '#44BB99'     # Blue-green
}

FRIENDLY_NAMES = {
    'none_none': 'static',
    'random_none': 'random',
    'biased_same': 'local (similar)',
    'biased_diff': 'local (opposite)',
    'bridge_same': 'bridge (similar)',
    'bridge_diff': 'bridge (opposite)',
    'wtf_none': 'wtf',
    'node2vec_none': 'node2vec'
}

# Data cache to avoid repeated file reads
class DataCache:
    """Simple cache for trajectory data to avoid re-reading files."""
    
    def __init__(self, max_size=5):
        """Initialize with maximum number of cached items."""
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key):
        """Get data from cache if available."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key, data):
        """Store data in cache, removing least recently used if full."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = data
        self.access_times[key] = time.time()

# Initialize global cache
trajectory_cache = DataCache(max_size=10)

def setup_plotting_style():
    """Configure plotting style to match paper."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.figsize': (20, 12),
        'figure.dpi': 300
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        "axes.grid": True,
        "grid.color": 'white',
        'grid.linestyle': 'solid', 
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.5,
        "axes.spines.bottom": True,
        "grid.alpha": 0.8,
        "xtick.bottom": True,
        "ytick.left": True
    })

def get_data_file():
    """Get the data file path from user input."""
    file_list = [f for f in os.listdir("../Output") 
                 if f.endswith(".csv") and "param_sweep" in f]
    
    if not file_list:
        print("No parameter sweep files found in the Output directory.")
        sys.exit(1)
    
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    try:
        file_index = int(input("Enter the index of the file you want to plot: "))
        if 0 <= file_index < len(file_list):
            return os.path.join("../Output", file_list[file_index])
        else:
            print("Invalid index. Using the first file.")
            return os.path.join("../Output", file_list[0])
    except ValueError:
        print("Invalid input. Using the first file.")
        return os.path.join("../Output", file_list[0])

def find_inflection(seq):
    """Calculate inflection point in opinion trajectory."""
    # Apply gaussian filter to smooth the trajectory
    smooth = gaussian_filter1d(seq, 600)  # Using original sigma value
    
    # Calculate second derivative
    d2 = np.gradient(np.gradient(smooth))
    
    # Find where second derivative changes sign
    infls = np.where(np.diff(np.sign(d2)))[0]
    
    # Set minimum threshold for inflection point
    inf_min = 5000
    
    # Find the first inflection point after the minimum threshold
    for i in infls:
        if i >= inf_min and i < 20000:
            return i
    
    return False

def estimate_convergence_rate(trajec, loc=None, regwin=10):
    """Estimate convergence rate around specified location using linear regression."""
    if loc is None or loc < regwin or loc + regwin >= len(trajec):
        return 0
    
    # Setup x and y arrays for regression window
    x = np.arange(loc-regwin, loc+regwin+1)
    y = trajec[loc-regwin: loc+regwin+1]
    
    # Ensure x and y are valid
    if len(x) < 3:  # Need at least 3 points for meaningful regression
        return 0
    
    # Linear regression
    n = len(x)
    mx, my = np.mean(x), np.mean(y)
    ssxy = np.sum(y*x) - n*my*mx
    ssxx = np.sum(x*x) - n*mx*mx
    
    # Calculate slope
    if ssxx == 0:
        return 0
    b1 = ssxy / ssxx
    
    # Calculate convergence rate
    denominator = abs(trajec[loc] - 1)
    if denominator < 0.001:
        rate = -b1 / 0.001
    else:
        rate = -b1 / denominator
    
    return rate

def get_friendly_scenario_name(scenario, rewiring):
    """Convert scenario name to friendly name based on its rewiring mode."""
    # Handle NaN values
    if pd.isna(scenario):
        return "Unknown"
    if pd.isna(rewiring):
        rewiring = "none"
    
    # Convert to string and lowercase for consistency
    scenario = str(scenario).lower()
    rewiring = str(rewiring).lower()
    
    if scenario == "none":
        key = "none_none"
    elif scenario == "random":
        key = "random_none"
    elif scenario == "wtf":
        key = "wtf_none"
    elif scenario == "node2vec":
        key = "node2vec_none"
    elif scenario in ["biased", "bridge"]:
        key = f"{scenario}_{rewiring}"
    else:
        key = f"{scenario}_none"
    
    return FRIENDLY_NAMES.get(key, f"{scenario} ({rewiring})")

def analyze_data_structure(filepath):
    """Analyze the structure of the data file to determine scenarios, topologies, etc."""
    print(f"Analyzing data structure of {filepath}...")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    # First pass: determine column names and structure
    sample = pd.read_csv(filepath, nrows=5)
    column_names = sample.columns.tolist()
    
    # Look for specific columns we need
    required_columns = ['scenario', 'type', 'rewiring', 'polarisingNode_f', 't', 'avg_state', 'model_run']
    missing_columns = [col for col in required_columns if col not in column_names]
    
    if missing_columns:
        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
        print(f"Available columns: {', '.join(column_names)}")
        sys.exit(1)
    
    # Extract unique values efficiently using SQL-like queries
    query_columns = ['scenario', 'type', 'rewiring', 'polarisingNode_f']
    unique_values = {}
    
    for col in query_columns:
        if col in column_names:
            # Use SQL to get unique values
            try:
                unique_query = f"SELECT DISTINCT [{col}] FROM filepath"
                unique_values[col] = pd.read_csv(filepath, usecols=[col]).dropna()[col].unique()
            except Exception as e:
                print(f"Error reading unique values for {col}: {e}")
                unique_values[col] = []
    
    scenarios = sorted([str(s) for s in unique_values.get('scenario', [])])
    topologies = sorted([str(t) for t in unique_values.get('type', [])])
    
    # Extract rewiring modes for each scenario
    rewiring_modes = {}
    scenario_rewiring_pairs = pd.read_csv(
        filepath, 
        usecols=['scenario', 'rewiring'] if 'rewiring' in column_names else ['scenario']
    ).dropna().drop_duplicates()
    
    for _, row in scenario_rewiring_pairs.iterrows():
        scenario = row['scenario']
        rewiring = row.get('rewiring', 'none')  # Default to 'none' if column doesn't exist
        rewiring_modes[scenario] = rewiring
    
    # Extract parameter values
    param_values = {}
    for param in ['polarisingNode_f']:
        if param in column_names:
            param_values[param] = sorted(pd.read_csv(filepath, usecols=[param]).dropna()[param].unique())
    
    print(f"Found {len(scenarios)} scenarios, {len(topologies)} topologies")
    print(f"Parameters: {list(param_values.keys())}")
    print(f"Topologies: {topologies}")
    print(f"Scenarios: {scenarios}")
    
    return scenarios, topologies, rewiring_modes, param_values

def extract_trajectories(filepath, topology, scenario, param_name, param_value, 
                        time_range=None, max_runs=100):
    """
    Extract trajectories for a specific topology, scenario, and parameter value.
    
    Parameters:
    - filepath: Path to the data file
    - topology: Network topology to extract data for
    - scenario: Rewiring scenario to extract data for
    - param_name: Parameter name for the sweep
    - param_value: Specific parameter value to extract data for
    - time_range: Optional tuple (start_time, end_time) to filter trajectories
    - max_runs: Maximum number of runs to extract
    """
    # Create cache key
    cache_key = f"{filepath}_{topology}_{scenario}_{param_name}_{param_value}_{time_range}"
    
    # Check cache first
    cached_data = trajectory_cache.get(cache_key)
    if cached_data is not None:
        return cached_data
    
    print(f"Extracting trajectories for {topology}, {scenario}, {param_name}={param_value}")
    
    # Prepare query conditions
    conditions = []
    conditions.append(f"type == '{topology}'")
    conditions.append(f"scenario == '{scenario}'")
    conditions.append(f"{param_name} == {param_value}")
    
    # Add time conditions if provided
    if time_range:
        start_time, end_time = time_range
        if start_time is not None:
            conditions.append(f"t >= {start_time}")
        if end_time is not None:
            conditions.append(f"t <= {end_time}")
    
    # Combine conditions
    query = " and ".join(conditions)
    
    # Read only needed columns
    columns = ['t', 'avg_state', 'model_run', param_name, 'type', 'scenario']
    
    try:
        # Read data efficiently with vectorized operations
        filtered_data = pd.read_csv(filepath, usecols=columns).query(query)
    except Exception as e:
        print(f"Error reading data: {e}")
        return {}
    
    # Early return if no data
    if filtered_data.empty:
        print(f"No data found for {topology}, {scenario}, {param_name}={param_value}")
        return {}
    
    # Get unique run IDs and limit to max_runs
    run_ids = filtered_data['model_run'].unique()
    
    if len(run_ids) > max_runs:
        # Use only the first max_runs IDs for consistency
        run_ids = sorted(run_ids)[:max_runs]
    
    # Extract trajectories for selected runs
    trajectories = {}
    
    for run_id in run_ids:
        run_data = filtered_data[filtered_data['model_run'] == run_id]
        
        # Sort by timestep
        run_data = run_data.sort_values('t')
        
        # Skip if not enough data points for inflection analysis
        if len(run_data) < 5000:  # Need enough points for inflection (minimum threshold)
            continue
        
        # Extract time and state
        times = run_data['t'].values
        states = run_data['avg_state'].values
        
        # Store as trajectory
        trajectories[run_id] = (times, states)
    
    # Store in cache
    trajectory_cache.set(cache_key, trajectories)
    
    print(f"Extracted {len(trajectories)} trajectories")
    return trajectories

def calculate_convergence_rates(trajectories):
    """Calculate convergence rates for a set of trajectories using optimized vectorized operations."""
    rates = []
    
    for run_id, (times, states) in trajectories.items():
        # Find inflection point
        inflection_idx = find_inflection(states)
        
        # Calculate rate if inflection point found
        if inflection_idx:
            rate = estimate_convergence_rate(states, loc=inflection_idx)
            if rate != 0:
                rates.append(rate * 1000)  # Scale for visualization
    
    return rates

def process_scenario(args):
    """Process a single scenario for efficient parallel execution."""
    filepath, topology, scenario, param_name, param_values, time_range = args
    
    print(f"Processing {topology} - {scenario}")
    
    results = {
        'rates': {},
        'distributions': {},
        'param_values': param_values
    }
    
    for val in param_values:
        # Extract trajectories
        trajectories = extract_trajectories(
            filepath, topology, scenario, param_name, val, time_range
        )
        
        # Calculate rates
        rates = calculate_convergence_rates(trajectories)
        
        if rates:
            results['rates'][val] = np.median(rates)
            results['distributions'][val] = rates
            
    return scenario, results

def create_heatmap_for_topology(topology, valid_scenarios, results, rewiring_modes):
    """Create heatmap visualization for a single topology with all valid scenarios."""
    print(f"Creating heatmap for topology: {topology}")
    
    # Determine grid layout
    n_scenarios = len(valid_scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), 
                           squeeze=False)
    
    # Find global min/max rates for consistent scale
    all_rates = []
    for scenario in valid_scenarios:
        for rates in results[scenario]['distributions'].values():
            all_rates.extend(rates)
    
    if all_rates:
        rate_min = np.percentile(all_rates, 1)  # 1st percentile to exclude outliers
        rate_max = np.percentile(all_rates, 99)  # 99th percentile
    else:
        rate_min, rate_max = 0, 1
    
    print(f"Rate range: {rate_min:.2f} to {rate_max:.2f}")
    
    # Number of bins for the heatmap
    rate_bins = 20
    
    # Process each scenario
    for i, scenario in enumerate(valid_scenarios):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        
        # Get friendly name
        friendly_name = get_friendly_scenario_name(scenario, rewiring_modes.get(scenario, 'none'))
        
        # Get data for this scenario
        scenario_data = results[scenario]
        param_vals = scenario_data['param_values']
        
        # Create 2D histogram data
        hist_data = np.zeros((len(param_vals), rate_bins))
        rate_bin_edges = np.linspace(rate_min, rate_max, rate_bins + 1)
        
        # Fill histogram
        for j, val in enumerate(param_vals):
            if val in scenario_data['distributions']:
                rates = scenario_data['distributions'][val]
                if rates:
                    hist, _ = np.histogram(rates, bins=rate_bin_edges)
                    if np.sum(hist) > 0:
                        hist = hist / np.sum(hist)  # Normalize
                    hist_data[j, :] = hist
        
        # Plot heatmap
        im = ax.imshow(
            hist_data.T,
            aspect='auto',
            origin='lower',
            extent=[0, len(param_vals)-1, rate_min, rate_max],
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Count')
        
        # Set title and labels with algorithm-specific color
        title_color = FRIENDLY_COLORS.get(friendly_name, 'black')
        ax.set_title(friendly_name, color=title_color, fontsize=FONT_SIZE+2, fontweight='bold')
        ax.set_xlabel('Polarizing Node Fraction', fontsize=FONT_SIZE)
        ax.set_ylabel('Convergence Rate (×10³)', fontsize=FONT_SIZE)
        
        # Set x-ticks to parameter values
        x_ticks = np.arange(len(param_vals))
        x_tick_labels = [f'{val:.2f}' for val in param_vals]
        tick_spacing = max(1, len(param_vals) // 5)
        ax.set_xticks(x_ticks[::tick_spacing])
        ax.set_xticklabels(x_tick_labels[::tick_spacing])
        
        # Add median line
        medians = []
        for j, val in enumerate(param_vals):
            if val in scenario_data['rates']:
                medians.append((j, scenario_data['rates'][val]))
        
        if medians:
            x_med, y_med = zip(*medians)
            ax.plot(x_med, y_med, 'r-', linewidth=2, label='Median')
            ax.legend(loc='best')
    
    # Hide unused subplots
    for i in range(len(valid_scenarios), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].set_visible(False)
    
    # Add overall title
    plt.suptitle(f'Convergence Rate Distributions - {topology.upper()}', 
                fontsize=FONT_SIZE+4, y=0.98, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    for ext in ['pdf', 'png']:
        save_path = f'../Figs/ConvergenceRates/convergence_rate_{topology}.{ext}'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"Saved heatmap for topology {topology}")

def create_convergence_heatmaps(filepath, scenarios, topologies, rewiring_modes, param_values, 
                              param_name='polarisingNode_f', time_range=None,
                              n_processes=None):
    """Create heatmaps showing convergence rate distributions for each topology and scenario."""
    print("Creating convergence rate heatmaps...")
    
    # Create output directory
    os.makedirs('../Figs/ConvergenceRates', exist_ok=True)
    
    # Set number of processes
    if n_processes is None:
        n_processes = min(multiprocessing.cpu_count() - 1, 8)  # Leave one core free
    
    print(f"Using {n_processes} parallel processes")
    
    # Process each topology
    for topology in topologies:
        print(f"\nProcessing topology: {topology}")
        
        # Store results for this topology
        all_results = {}
        
        # Quick check to find valid scenarios with data
        valid_scenarios = []
        
        # Only test one parameter value to see if there's any data
        test_param_value = param_values[param_name][0]
        for scenario in scenarios:
            trajectories = extract_trajectories(
                filepath, topology, scenario, param_name, test_param_value, time_range, max_runs=1
            )
            
            if trajectories:
                valid_scenarios.append(scenario)
        
        if not valid_scenarios:
            print(f"No data for topology {topology}, skipping")
            continue
        
        print(f"Found data for {len(valid_scenarios)} scenarios: {valid_scenarios}")
        
        # Prepare arguments for parallel processing
        args_list = [
            (filepath, topology, scenario, param_name, param_values[param_name], time_range)
            for scenario in valid_scenarios
        ]
        
        # Process scenarios in parallel
        start_time = time.time()
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.map(process_scenario, args_list)
        
        # Collect results
        for scenario, scenario_results in results:
            all_results[scenario] = scenario_results
        
        end_time = time.time()
        print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
        
        # Create heatmap
        create_heatmap_for_topology(topology, valid_scenarios, all_results, rewiring_modes)

def main():
    """Main execution function to create heatmaps for all topologies."""
    # Setup plotting style
    setup_plotting_style()
    
    # Get data file path - Modify this path to point to your file
    data_path = get_data_file()  # Interactive file selection
    
    # Analyze data structure
    scenarios, topologies, rewiring_modes, param_values = analyze_data_structure(data_path)
    
    # CUSTOMIZE HERE: Filter to specific topologies and scenarios
    # Comment out these lines to process all detected topologies/scenarios
    topologies = ["cl", "FB"]  # Only process these topologies
    scenarios = ["random", "biased"]  # Only process these scenarios
    
    # CUSTOMIZE HERE: Set time range for analysis (in timesteps)
    # Set to None to use all available timesteps
    time_range = (5000, 30000)  # Only analyze between timesteps 5000 and 30000
    
    # CUSTOMIZE HERE: Set number of parallel processes
    # Set to 1 if having issues with multiprocessing in Spyder
    n_processes = max(1, multiprocessing.cpu_count() - 1)  # Use all but one core
    
    print(f"Processing topologies: {topologies}")
    print(f"Processing scenarios: {scenarios}")
    print(f"Time range: {time_range}")
    print(f"Using {n_processes} processes")
    
    # Create convergence rate heatmaps for selected topologies and scenarios
    create_convergence_heatmaps(
        data_path, scenarios, topologies, rewiring_modes, param_values,
        time_range=time_range, n_processes=n_processes
    )
    
    print("Processing complete. All heatmaps created successfully.")

if __name__ == "__main__":
    # Track and print total execution time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")