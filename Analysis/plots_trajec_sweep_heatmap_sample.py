#!/usr/bin/env python3
"""
Memory-optimized heatmap generator for parameter sweep visualization using Polars.
Designed to handle very large CSV files (50GB+) by using lazy evaluation,
streaming processing, and aggressive memory management.
"""

import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import gc
import sys
import time
import traceback
import random
from functools import lru_cache
import psutil  # For memory monitoring
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Suppress warnings about fragmented DataFrames
warnings.filterwarnings("ignore")

# ====================== CONFIGURATION ======================
# Optimized settings for large files

# Input file - change this to your CSV file path
INPUT_FILE = "../Output/param_sweep_individual_N_789_polarisingNode_f_0.0_1.0_10_2025-03-07.csv"

# Output directory
OUTPUT_DIR = "../Figs/ConvergenceRates"

# Sampling parameters - optimized for large files
SAMPLE_FRACTION = 0.05  # Sample 5% of runs for analysis
CHUNK_SIZE = 50000     # Process in larger chunks with Polars for better performance
MAX_SAMPLES_PER_COMBINATION = 50  # Maximum number of runs to analyze per parameter combination

# Time range for analysis (in timesteps)
TIME_RANGE = (5000, 30000)  # (start_time, end_time)

# Parameter to analyze
PARAM_NAME = "polarisingNode_f"

# Plot settings
FONT_SIZE = 14
RANDOM_SEED = 42

# Cache sizes - reduced for large files
INFLECTION_CACHE_SIZE = 128
CONVERGENCE_CACHE_SIZE = 128

# Scenario names and colors
FRIENDLY_COLORS = {
    'static': '#EE7733',
    'random': '#0077BB',
    'local (similar)': '#33BBEE',
    'local (opposite)': '#009988',
    'bridge (similar)': '#CC3311',
    'bridge (opposite)': '#EE3377',
    'wtf': '#BBBBBB',
    'node2vec': '#44BB99'
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

# ====================== MEMORY MONITORING ======================

def print_memory_usage(message=""):
    """Print current memory usage with optional message."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage {message}: {mem_info.rss / (1024**3):.2f} GB")

# ====================== FUNCTIONS ======================

def get_data_file():
    """Get the data file path from user input or command line argument."""
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
        
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
    
def setup_plotting():
    """Configure plotting style."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'figure.figsize': (20, 12),
        'figure.dpi': 300
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set_style("ticks")

@lru_cache(maxsize=INFLECTION_CACHE_SIZE)
def find_inflection(seq_tuple):
    """Calculate inflection point in opinion trajectory."""
    seq = np.array(seq_tuple)
    
    if len(seq) < 7000:
        return False
    
    # Smooth trajectory and find inflection points
    try:
        smooth = gaussian_filter1d(seq, 600)
        d2 = np.gradient(np.gradient(smooth))
        infls = np.where(np.diff(np.sign(d2)))[0]
    except Exception as e:
        print(f"Error in filtering: {e}")
        return False
    
    # Find valid inflection points for depolarization
    inf_min = 5000
    for i in infls:
        if i >= inf_min and i < 20000:
            window_size = min(100, len(seq) - i - 1)
            if window_size > 10 and np.polyfit(range(window_size), seq[i:i+window_size], 1)[0] > 0:
                return i
    
    return False

@lru_cache(maxsize=CONVERGENCE_CACHE_SIZE)
def estimate_convergence_rate(trajec_tuple, loc=None, regwin=10):
    """Estimate convergence rate using linear regression."""
    trajec = np.array(trajec_tuple)
    
    if loc is None or loc < regwin or loc + regwin >= len(trajec):
        return 0
    
    # Calculate regression
    x = np.arange(loc-regwin, loc+regwin+1)
    y = trajec[loc-regwin: loc+regwin+1]
    
    if len(x) < 3:
        return 0
    
    # Linear regression
    n = len(x)
    mx, my = np.mean(x), np.mean(y)
    ssxy = np.sum(y*x) - n*my*mx
    ssxx = np.sum(x*x) - n*mx*mx
    
    if ssxx == 0:
        return 0
    b1 = ssxy / ssxx
    
    # Only return positive rates (depolarization)
    if b1 <= 0:
        return 0
        
    denominator = abs(trajec[loc] - 1)
    if denominator < 0.001:
        rate = b1 / 0.001
    else:
        rate = b1 / denominator
    
    return rate

def get_friendly_name(scenario, rewiring):
    """Get user-friendly algorithm name."""
    if scenario is None:
        return "Unknown"
    
    if rewiring is None:
        rewiring = "none"
    
    scenario = str(scenario).lower()
    rewiring = str(rewiring).lower()
    
    key = f"{scenario}_{rewiring}"
    if scenario in ["none", "random", "wtf", "node2vec"]:
        key = f"{scenario}_none"
    
    return FRIENDLY_NAMES.get(key, f"{scenario} ({rewiring})")

def get_schema_info(filepath):
    """Get schema information from CSV file efficiently."""
    print(f"Getting schema from {filepath}...")
    try:
        # Use Polars LazyFrame to inspect the schema without loading the data
        schema = pl.scan_csv(filepath, n_rows=10).collect().schema
        dtypes = {name: str(dtype) for name, dtype in schema.items()}
        print(f"Schema: {dtypes}")
        return list(schema.keys()), dtypes
    except Exception as e:
        print(f"Error getting schema: {e}")
        # Try a more direct approach if scanning fails
        try:
            headers = pl.read_csv(filepath, n_rows=1).columns
            print(f"Headers: {headers}")
            return headers, {}
        except Exception as e2:
            print(f"Error reading headers: {e2}")
            return [], {}

def estimate_row_count(filepath):
    """Estimate number of rows in a large CSV file without reading it entirely."""
    print(f"Estimating row count for {filepath}...")
    
    file_size = os.path.getsize(filepath)
    
    # For very large files, estimate based on sampling
    if file_size > 1e9:  # > 1GB
        try:
            # Read first and last MB to estimate line counts
            line_size_samples = []
            
            # Sample the beginning
            with open(filepath, 'rb') as f:
                sample = f.read(1024 * 1024)  # 1MB
                lines = sample.count(b'\n')
                line_size_samples.append(len(sample) / max(1, lines))
            
            # Sample the middle
            with open(filepath, 'rb') as f:
                f.seek(file_size // 2)
                # Find the next newline
                sample_line = f.readline()
                sample = f.read(1024 * 1024)  # 1MB
                lines = sample.count(b'\n')
                line_size_samples.append(len(sample) / max(1, lines))
            
            # Calculate average line size
            avg_line_size = sum(line_size_samples) / len(line_size_samples)
            estimated_lines = int(file_size / avg_line_size)
            
            print(f"Estimated {estimated_lines:,} lines based on sampling")
            return estimated_lines
            
        except Exception as e:
            print(f"Error in estimation: {e}")
    
    # For smaller files, or if estimation fails
    try:
        with open(filepath, 'rb') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error counting lines: {e}")
        return 1000000  # Default assumption

def sample_unique_values(filepath, column, limit=20):
    """Sample unique values for a column in a large CSV file."""
    print(f"Sampling unique values for {column}...")
    
    try:
        # Use lazy evaluation to efficiently get unique values
        unique_vals = (
            pl.scan_csv(filepath)
            .select(pl.col(column).unique())
            .collect()
            .to_series()
            .to_list()
        )
        
        if column == PARAM_NAME:
            # Convert parameter values to float and sort them
            unique_vals = sorted([float(v) for v in unique_vals if v is not None])
        
        print(f"Found {len(unique_vals)} unique values for {column}")
        return unique_vals
    
    except Exception as e:
        print(f"Error sampling unique values: {e}")
        
        # Fallback to direct sampling for very large files
        try:
            # Sample from different parts of the file
            file_size = os.path.getsize(filepath)
            positions = [0, file_size // 3, 2 * file_size // 3]
            
            unique_values = set()
            for pos in positions:
                # Skip to position and read a chunk
                df = pl.read_csv(filepath, n_rows=10000, skip_rows=max(1, pos // 1000))
                if column in df.columns:
                    unique_values.update(df.select(pl.col(column)).to_series().unique().to_list())
            
            if column == PARAM_NAME:
                # Convert parameter values to float and sort them
                unique_values = sorted([float(v) for v in unique_values if v is not None])
            
            print(f"Found {len(unique_values)} unique values for {column} using fallback")
            return list(unique_values)
        
        except Exception as e2:
            print(f"Fallback sampling also failed: {e2}")
            return []

def find_topologies_and_scenarios(filepath):
    """Find all topologies and scenarios in the data."""
    print("Finding topologies and scenarios...")
    
    topologies = sample_unique_values(filepath, 'type')
    scenarios = sample_unique_values(filepath, 'scenario')
    
    print(f"Found topologies: {topologies}")
    print(f"Found scenarios: {scenarios}")
    
    return topologies, scenarios

def get_parameter_values(filepath, param_name=PARAM_NAME):
    """Get all unique parameter values."""
    print(f"Finding unique values for {param_name}...")
    
    param_values = sample_unique_values(filepath, param_name)
    print(f"Parameter values: {param_values}")
    
    return param_values

def get_rewiring_modes(filepath, scenarios):
    """Determine rewiring mode for each scenario."""
    print("Finding rewiring modes...")
    
    rewiring_modes = {}
    
    # Default values for known scenarios
    for scenario in scenarios:
        scenario_lower = str(scenario).lower()
        if scenario_lower in ['none', 'random', 'wtf', 'node2vec']:
            rewiring_modes[scenario] = 'none'
        else:
            rewiring_modes[scenario] = 'diff'  # Default
    
    # Try to determine actual rewiring modes from data
    try:
        # Sample data to check rewiring modes
        query = (
            pl.scan_csv(filepath)
            .filter(pl.col("scenario").is_in(scenarios))
            .select(["scenario", "rewiring"])
            .collect()
        )
        
        # Group by scenario and find most common rewiring mode
        for scenario in scenarios:
            scenario_data = query.filter(pl.col("scenario") == scenario)
            if len(scenario_data) > 0:
                # Count occurrence of each rewiring mode
                counts = (
                    scenario_data
                    .group_by("rewiring")
                    .agg(pl.count())
                    .sort("count", descending=True)
                )
                
                if len(counts) > 0:
                    rewiring_modes[scenario] = counts[0, "rewiring"]
        
    except Exception as e:
        print(f"Error determining rewiring modes: {e}")
    
    print(f"Rewiring modes: {rewiring_modes}")
    return rewiring_modes

def find_valid_scenarios(filepath, topology, scenarios):
    """Find scenarios that exist for a specific topology."""
    print(f"Finding valid scenarios for {topology}...")
    
    valid_scenarios = []
    
    try:
        # Query for scenarios that exist for this topology
        query = (
            pl.scan_csv(filepath)
            .filter(pl.col("type") == topology)
            .select("scenario")
            .unique()
            .collect()
        )
        
        # Check which scenarios from our list exist
        existing_scenarios = set(query["scenario"].to_list())
        valid_scenarios = [s for s in scenarios if s in existing_scenarios]
        
        print(f"Found {len(valid_scenarios)} valid scenarios for {topology}")
        
    except Exception as e:
        print(f"Error finding valid scenarios: {e}")
        # Default to all scenarios if query fails
        valid_scenarios = scenarios
    
    return valid_scenarios

def extract_run_trajectories(filepath, topology, scenario, param_value, time_range):
    """Extract trajectories for a specific combination using Polars."""
    print(f"Extracting trajectories for {topology}/{scenario}, param={param_value}")
    
    # Set time constraints
    start_time, end_time = time_range
    
    try:
        # Use Polars LazyFrame for efficient filtering
        query = (
            pl.scan_csv(filepath)
            .filter(
                (pl.col("type") == topology) & 
                (pl.col("scenario") == scenario) & 
                (pl.col(PARAM_NAME) == param_value) &
                (pl.col("t") >= start_time) & 
                (pl.col("t") <= end_time)
            )
        )
        
        # Get unique run IDs
        run_ids = query.select("model_run").unique().collect()["model_run"].to_list()
        
        # Randomly sample run IDs
        random.seed(RANDOM_SEED)
        num_samples = min(MAX_SAMPLES_PER_COMBINATION, int(len(run_ids) * SAMPLE_FRACTION))
        sampled_run_ids = random.sample(run_ids, max(1, num_samples))
        
        print(f"Sampled {len(sampled_run_ids)} of {len(run_ids)} runs")
        
        # Extract trajectories for sampled runs
        all_rates = []
        
        # Process runs in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            # Submit trajectory extraction tasks
            futures = []
            for run_id in sampled_run_ids:
                futures.append(
                    executor.submit(
                        process_single_run, 
                        filepath, topology, scenario, param_value, 
                        run_id, time_range
                    )
                )
            
            # Collect results
            for future in futures:
                rate = future.result()
                if rate > 0:
                    all_rates.append(rate * 1000)  # Scale for visualization
        
        return all_rates
        
    except Exception as e:
        print(f"Error extracting trajectories: {e}")
        traceback.print_exc()
        return []

def process_single_run(filepath, topology, scenario, param_value, run_id, time_range):
    """Process a single run to calculate convergence rate."""
    try:
        # Extract data for this run
        run_data = (
            pl.scan_csv(filepath)
            .filter(
                (pl.col("type") == topology) & 
                (pl.col("scenario") == scenario) & 
                (pl.col(PARAM_NAME) == param_value) &
                (pl.col("model_run") == run_id) &
                (pl.col("t") >= time_range[0]) & 
                (pl.col("t") <= time_range[1])
            )
            .select(["t", "avg_state"])
            .sort("t")
            .collect()
        )
        
        # Check if we have enough data points
        if len(run_data) < 1000:
            return 0
            
        # Convert to numpy array for analysis
        states = run_data["avg_state"].to_numpy()
        states_tuple = tuple(states)
        
        # Find inflection point
        inflection_idx = find_inflection(states_tuple)
        
        if inflection_idx:
            # Calculate convergence rate
            rate = estimate_convergence_rate(states_tuple, loc=inflection_idx)
            return rate
        
        return 0
        
    except Exception as e:
        print(f"Error processing run {run_id}: {e}")
        return 0

def process_topology(filepath, topology, scenarios, param_values, rewiring_modes):
    """Process all scenarios for a topology using Polars for efficiency."""
    print(f"\nProcessing topology: {topology}")
    print_memory_usage(f"before processing {topology}")
    
    # Find valid scenarios for this topology
    valid_scenarios = find_valid_scenarios(filepath, topology, scenarios)
    
    if not valid_scenarios:
        print(f"No valid scenarios found for topology {topology}")
        return {}
    
    # Initialize results
    results = {}
    
    # Process each scenario for this topology
    for scenario in valid_scenarios:
        print(f"Processing scenario: {scenario}")
        
        scenario_results = {
            'rates': {},
            'distributions': {},
            'param_values': param_values
        }
        
        # Process each parameter value
        for param_value in param_values:
            print(f"  Parameter {PARAM_NAME}={param_value}")
            
            try:
                rates = extract_run_trajectories(
                    filepath, topology, scenario, param_value, TIME_RANGE
                )
                
                if rates:
                    scenario_results['rates'][param_value] = np.median(rates)
                    scenario_results['distributions'][param_value] = rates
                    print(f"    Found {len(rates)} valid rates, median: {np.median(rates):.4f}")
                else:
                    print(f"    No valid rates found")
                    
            except Exception as e:
                print(f"Error processing {scenario}/{param_value}: {e}")
                traceback.print_exc()
            
            # Force garbage collection after each parameter
            gc.collect()
        
        # Store results if we found any data
        if any(scenario_results['distributions']):
            results[scenario] = scenario_results
        
        # Clear memory
        print_memory_usage(f"after processing {scenario}")
    
    return results

def create_heatmap(topology, results, rewiring_modes):
    """Generate heatmap visualization for convergence rates."""
    print(f"Creating heatmap for {topology}")
    
    # Get list of valid scenarios
    valid_scenarios = [s for s in results.keys() 
                     if results[s]['distributions']]  # Only include scenarios with data
    
    if not valid_scenarios:
        print("No valid data to plot")
        return
    
    # Determine grid layout
    n_scenarios = len(valid_scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    
    # Find global min/max rates for consistent scale
    all_rates = []
    for scenario in valid_scenarios:
        for rates in results[scenario]['distributions'].values():
            if len(all_rates) < 10000:  # Limit sample size for memory
                all_rates.extend(rates[:min(len(rates), 100)])  # Take at most 100 values from each
    
    if all_rates:
        rate_min = np.percentile(all_rates, 1)  # 1st percentile
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
        
        # Get friendly name and color
        friendly_name = get_friendly_name(scenario, rewiring_modes.get(scenario, 'none'))
        title_color = FRIENDLY_COLORS.get(friendly_name, 'black')
        
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
        
        # Set title and labels
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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ['pdf', 'png']:
        save_path = f'{OUTPUT_DIR}/convergence_rate_{topology}.{ext}'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"Saved heatmap for {topology}")
    gc.collect()

def main():
    """Main execution function."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    setup_plotting()
    
    # Get input file path
    input_file = get_data_file() if INPUT_FILE == "../Output/param_sweep_individual_N_789_polarisingNode_f_0.0_1.0_10_2025-03-07.csv" else INPUT_FILE
    
    try:
        start_time = time.time()
        print(f"Processing file: {input_file}")
        print_memory_usage("at start")
        
        # Get file schema
        headers, dtypes = get_schema_info(input_file)
        if not headers:
            print("Error: Could not read schema from file")
            return
        
        # Get topologies and scenarios
        topologies, scenarios = find_topologies_and_scenarios(input_file)
        
        # Get parameter values
        param_values = get_parameter_values(input_file, PARAM_NAME)
        
        # Get rewiring modes
        rewiring_modes = get_rewiring_modes(input_file, scenarios)
        
        # Process each topology
        for topology in topologies:
            results = process_topology(
                input_file, topology, scenarios,
                param_values, rewiring_modes
            )
            
            if results:
                create_heatmap(topology, results, rewiring_modes)
            
            # Clear memory
            results = {}
            gc.collect()
            print_memory_usage(f"after processing topology {topology}")
        
        end_time = time.time()
        print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()