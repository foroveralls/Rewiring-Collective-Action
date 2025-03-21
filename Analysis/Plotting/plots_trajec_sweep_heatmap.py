#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved heatmap generator for parameter sweeps.
Creates one heatmap per topology with all rewiring algorithms.
Based on the original script with incremental improvements.
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
import traceback

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
    smooth = gaussian_filter1d(seq, 600)  # Original value
    
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
    if len(x) < 3:  # Need at least 3 points for regression
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
    
    # First pass: determine column names
    sample = pd.read_csv(filepath, nrows=5)
    column_names = sample.columns.tolist()
    
    # Look for specific columns we need
    required_columns = ['scenario', 'type', 'rewiring', 'polarisingNode_f', 't', 'avg_state', 'model_run']
    missing_columns = [col for col in required_columns if col not in column_names]
    
    if missing_columns:
        print(f"Warning: Missing columns: {', '.join(missing_columns)}")
    
    # Extract unique values efficiently by reading in small chunks
    chunk_size = 50000
    unique_values = {
        'scenario': set(),
        'type': set(),
        'rewiring': set(),
        'polarisingNode_f': set()
    }
    
    # Read in chunks to conserve memory
    for chunk in pd.read_csv(filepath, usecols=[col for col in unique_values.keys() if col in column_names], 
                           chunksize=chunk_size):
        for col in unique_values.keys():
            if col in chunk.columns:
                unique_values[col].update(chunk[col].dropna().unique())
    
    # Convert sets to sorted lists
    scenarios = sorted([str(s) for s in unique_values['scenario']])
    topologies = sorted([str(t) for t in unique_values['type']])
    
    # Extract rewiring modes for each scenario
    rewiring_modes = {}
    rewiring_values = list(unique_values['rewiring'])
    
    # Process small chunks to find scenario-rewiring pairs
    for chunk in pd.read_csv(filepath, usecols=['scenario', 'rewiring'] if 'rewiring' in column_names else ['scenario'], 
                           chunksize=chunk_size):
        pairs = chunk.dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            scenario = row['scenario']
            rewiring = row.get('rewiring', 'none')  # Default to 'none' if column doesn't exist
            rewiring_modes[scenario] = rewiring
    
    # Extract parameter values
    param_values = {}
    for param in ['polarisingNode_f']:
        if param in column_names:
            param_values[param] = sorted(unique_values[param])
    
    print(f"Found {len(scenarios)} scenarios, {len(topologies)} topologies")
    print(f"Parameters: {list(param_values.keys())}")
    print(f"Topologies: {topologies}")
    print(f"Scenarios: {scenarios}")
    
    return scenarios, topologies, rewiring_modes, param_values

def extract_trajectories(filepath, topology, scenario, param_name, param_value, time_range=None, max_runs=100):
    """Extract trajectories for a specific topology, scenario, and parameter value."""
    print(f"Extracting trajectories for {topology}, {scenario}, {param_name}={param_value}")
    
    # Read data in chunks to reduce memory usage
    chunk_size = 500000
    
    # Prepare query conditions
    query_str = f"type == '{topology}' and scenario == '{scenario}' and {param_name} == {param_value}"
    
    # Add time conditions if provided
    if time_range:
        start_time, end_time = time_range
        if start_time is not None:
            query_str += f" and t >= {start_time}"
        if end_time is not None:
            query_str += f" and t <= {end_time}"
    
    # Read only needed columns
    columns = ['t', 'avg_state', 'model_run', param_name, 'type', 'scenario']
    
    # Dictionary to store trajectories
    trajectories = {}
    run_count = 0
    
    try:
        # Process the file in chunks to reduce memory usage
        for chunk in pd.read_csv(filepath, usecols=columns, chunksize=chunk_size):
            # Filter chunk
            filtered_chunk = chunk.query(query_str)
            
            if filtered_chunk.empty:
                continue
                
            # Get unique run IDs in this chunk
            chunk_run_ids = filtered_chunk['model_run'].unique()
            
            # Process each run
            for run_id in chunk_run_ids:
                # Skip if we already have enough runs
                if run_count >= max_runs and run_id not in trajectories:
                    continue
                    
                # Get data for this run
                run_data = filtered_chunk[filtered_chunk['model_run'] == run_id]
                
                # Initialize trajectory if this is a new run
                if run_id not in trajectories:
                    trajectories[run_id] = ([], [])  # (times, states)
                    run_count += 1
                
                # Extract times and states
                times, states = trajectories[run_id]
                times.extend(run_data['t'].values)
                states.extend(run_data['avg_state'].values)
            
            # Force garbage collection after each chunk
            del filtered_chunk
            gc.collect()
    
    except Exception as e:
        print(f"Error reading data: {e}")
        return {}
    
    # Sort trajectories by time
    for run_id in list(trajectories.keys()):
        times, states = trajectories[run_id]
        
        # Skip if not enough data points
        if len(times) < 5000:  # Need enough points for inflection analysis
            del trajectories[run_id]
            continue
            
        # Sort by time
        sorted_indices = np.argsort(times)
        sorted_times = np.array(times)[sorted_indices]
        sorted_states = np.array(states)[sorted_indices]
        
        # Store sorted arrays
        trajectories[run_id] = (sorted_times, sorted_states)
    
    print(f"Extracted {len(trajectories)} trajectories")
    return trajectories

def calculate_convergence_rates(trajectories):
    """Calculate convergence rates for a set of trajectories."""
    rates = []
    count = 0
    
    for run_id, (times, states) in trajectories.items():
        try:
            # Find inflection point
            inflection_idx = find_inflection(states)
            
            # Calculate rate if inflection point found
            if inflection_idx:
                rate = estimate_convergence_rate(states, loc=inflection_idx)
                if rate != 0:
                    rates.append(rate * 1000)  # Scale for visualization
        except Exception as e:
            print(f"Error calculating rate for run {run_id}: {e}")
        
        # Periodically force garbage collection 
        count += 1
        if count % 10 == 0:
            gc.collect()
    
    return rates

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
    gc.collect()  # Force garbage collection after saving

def create_convergence_heatmaps(filepath, scenarios, topologies, rewiring_modes, param_values, 
                              param_name='polarisingNode_f', time_range=None):
    """Create heatmaps showing convergence rate distributions for each topology and scenario."""
    print("Creating convergence rate heatmaps...")
    
    # Create output directory
    os.makedirs('../Figs/ConvergenceRates', exist_ok=True)
    
    # Process each topology sequentially to minimize memory usage
    for topology in topologies:
        print(f"\nProcessing topology: {topology}")
        
        # Store results for this topology
        all_results = {}
        
        # Quick check to find valid scenarios with data
        valid_scenarios = []
        
        # Only test one parameter value to see if there's any data
        test_param_value = param_values[param_name][0]
        
        for scenario in scenarios:
            try:
                print(f"Checking if data exists for {topology}/{scenario}")
                trajectories = extract_trajectories(
                    filepath, topology, scenario, param_name, test_param_value, time_range, max_runs=1
                )
                
                if trajectories:
                    valid_scenarios.append(scenario)
                    print(f"Found data for {topology}/{scenario}")
                
                # Clear trajectories to free memory
                trajectories.clear()
                gc.collect()
            except Exception as e:
                print(f"Error checking data for {scenario}: {e}")
        
        if not valid_scenarios:
            print(f"No data for topology {topology}, skipping")
            continue
        
        print(f"Found data for {len(valid_scenarios)} scenarios: {valid_scenarios}")
        
        # Process each scenario sequentially
        for scenario in valid_scenarios:
            print(f"Processing scenario: {scenario}")
            
            try:
                # Process scenario
                scenario_results = {
                    'rates': {},
                    'distributions': {},
                    'param_values': param_values[param_name]
                }
                
                for val_idx, val in enumerate(param_values[param_name]):
                    print(f"  Parameter {param_name}={val} ({val_idx+1}/{len(param_values[param_name])})")
                    
                    # Extract trajectories
                    trajectories = extract_trajectories(
                        filepath, topology, scenario, param_name, val, time_range, max_runs=50
                    )
                    
                    # Calculate rates
                    rates = calculate_convergence_rates(trajectories)
                    
                    if rates:
                        scenario_results['rates'][val] = np.median(rates)
                        scenario_results['distributions'][val] = rates
                    
                    # Clear trajectories to free memory
                    trajectories.clear()
                    gc.collect()
                
                # Store results for this scenario
                all_results[scenario] = scenario_results
                
                # Force garbage collection after each scenario
                gc.collect()
            
            except Exception as e:
                print(f"Error processing scenario {scenario}: {e}")
                traceback.print_exc()
        
        # Create heatmap
        create_heatmap_for_topology(topology, valid_scenarios, all_results, rewiring_modes)
        
        # Clear results to free memory
        all_results.clear()
        valid_scenarios.clear()
        gc.collect()
        print(f"Finished processing topology: {topology}")

def main():
    """Main execution function to create heatmaps for all topologies."""
    # Setup plotting style
    setup_plotting_style()
    
    # Get data file path - Modify this path to point to your file
    data_path = get_data_file()  # Interactive file selection
   
    
    print(f"Processing file: {data_path}")
    print(f"File size: {os.path.getsize(data_path) / (1024 * 1024):.2f} MB")
    
    # Force garbage collection before starting
    gc.collect()
    
    # Analyze data structure
    scenarios, topologies, rewiring_modes, param_values = analyze_data_structure(data_path)
    
    print(scenarios, topologies, rewiring_modes)
    # CUSTOMIZE HERE: Filter to specific topologies and scenarios
    # Comment out these lines to process all detected topologies/scenarios
    topologies = ["cl"]  # Only process these topologies
    scenarios = ["random"]  # Only process these scenarios
    
    # CUSTOMIZE HERE: Set time range for analysis (in timesteps)
    # Set to None to use all available timesteps
    time_range = (5000, 30000)  # Only analyze between timesteps 5000 and 30000
    
    print(f"Processing topologies: {topologies}")
    print(f"Processing scenarios: {scenarios}")
    print(f"Time range: {time_range}")
    
    # Create convergence rate heatmaps for selected topologies and scenarios
    create_convergence_heatmaps(
        data_path, scenarios, topologies, rewiring_modes, param_values,
        time_range=time_range
    )
    
    # Final garbage collection
    gc.collect()

if __name__ == "__main__":
    # Track and print total execution time
    import traceback
    
    try:
        start_time = time.time()
        main()
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        # Save the crash report to a file
        crash_file = "heatmap_crash_report.txt"
        with open(crash_file, "w") as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        print(f"\nERROR: {str(e)}")
        print(f"Crash report saved to {crash_file}")
        print("Please check the crash report for details.")