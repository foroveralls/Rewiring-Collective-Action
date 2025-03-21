#!/usr/bin/env python3
"""
Memory-optimized convergence rate processor for parameter sweep analysis.
This script extracts convergence rates from trajectory data and saves the results for later visualization.
"""

import os
import numpy as np
import polars as pl
import gc
import sys
import time
import traceback
import random
from functools import lru_cache
import psutil
import warnings
import pickle
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# Suppress warnings
warnings.filterwarnings("ignore")

# ====================== CONFIGURATION ======================
# Control which topologies and scenarios to process
SELECTED_TOPOLOGIES = []  # Empty list means all topologies will be processed
SELECTED_SCENARIOS = []   # Empty list means all scenarios will be processed

# Input file - change this to your CSV file path
INPUT_FILE = "../Output/param_sweep_individual_N_789_polarisingNode_f_0.0_1.0_10_2025-03-07.csv"

# Output directory for saved results
OUTPUT_DIR = "../Output/ProcessedRates"

# Sampling parameters - optimized for large files
SAMPLE_FRACTION = 1.0  # Sample 100% of runs for analysis
CHUNK_SIZE = 500000    # Process in larger chunks with Polars for better performance
MAX_SAMPLES_PER_COMBINATION = 50  # Maximum number of runs to analyze per parameter combination

# Time range for analysis (in timesteps)
TIME_RANGE = (5000, 45000)  # (start_time, end_time)

# Parameter to analyze
PARAM_NAME = "polarisingNode_f"

# Analysis options
POSITIVE_RATES_ONLY = False  # Set to True to include only positive convergence rates

# Smoothing parameters for consistency
SMOOTHING_SIGMA = 300  # Gaussian smoothing parameter
MIN_SEQUENCE_LENGTH = 5000  # Minimum sequence length to analyze
INFLECTION_MIN_IDX = 5000  # Minimum index for inflection point
MAX_INFLECTION_FRACTION = 0.9  # Maximum position of inflection as fraction of sequence length

# Cache sizes - reduced for large files
INFLECTION_CACHE_SIZE = 128
CONVERGENCE_CACHE_SIZE = 128

# Random seed for reproducibility
RANDOM_SEED = 42

# ====================== MEMORY MANAGEMENT ======================

def limit_cpu_cores(max_cores):
    """Limit process to use only specified number of CPU cores"""
    process = psutil.Process()
    # Get list of available CPU cores (0, 1, 2, ...)
    all_cpus = list(range(psutil.cpu_count()))
    # Use only subset of cores
    cores_to_use = all_cpus[:max_cores]
    process.cpu_affinity(cores_to_use)
    print(f"Limited process to cores: {cores_to_use}")

def print_memory_usage(message=""):
    """Print current memory usage with optional message."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage {message}: {mem_info.rss / (1024**3):.2f} GB")

# ====================== DATA LOADING FUNCTIONS ======================

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
        file_index = int(input("Enter the index of the file you want to process: "))
        if 0 <= file_index < len(file_list):
            return os.path.join("../Output", file_list[file_index])
        else:
            print("Invalid index. Using the first file.")
            return os.path.join("../Output", file_list[0])
    except ValueError:
        print("Invalid input. Using the first file.")
        return os.path.join("../Output", file_list[0])

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
            # For biased and bridge, we need to check both 'same' and 'diff' modes
            rewiring_modes[scenario] = 'diff'  # Default
    
    # Try to determine actual rewiring modes from data
    try:
        # Sample data to check for distinct scenario-rewiring combinations
        query = (
            pl.scan_csv(filepath)
            .filter(pl.col("scenario").is_in(scenarios))
            .select(["scenario", "rewiring"])
            .unique()
            .collect()
        )
        
        # Create a mapping of all scenario-rewiring combinations found
        scenario_rewiring_map = {}
        for row in query.iter_rows(named=True):
            scenario = row["scenario"]
            rewiring = row["rewiring"]
            
            if scenario not in scenario_rewiring_map:
                scenario_rewiring_map[scenario] = []
            
            if rewiring and rewiring not in scenario_rewiring_map[scenario]:
                scenario_rewiring_map[scenario].append(rewiring)
        
        # Process scenario-rewiring map to create multiple entries when needed
        expanded_scenarios = []
        for scenario, rewirings in scenario_rewiring_map.items():
            # For 'none', 'random', 'wtf', 'node2vec', we only need one entry
            if str(scenario).lower() in ['none', 'random', 'wtf', 'node2vec']:
                rewiring_modes[scenario] = 'none'
                expanded_scenarios.append(scenario)
            else:
                # For 'biased' and 'bridge', create entries for each rewiring mode
                for rewiring in rewirings:
                    if rewiring:
                        # Create a "combined ID" to distinguish different modes of the same algorithm
                        combined_id = f"{scenario}_{rewiring}"
                        rewiring_modes[combined_id] = rewiring
                        expanded_scenarios.append(combined_id)
        
        # Update the scenarios list with expanded entries when appropriate
        print(f"Expanded scenarios with rewiring modes: {expanded_scenarios}")
        
    except Exception as e:
        print(f"Error determining rewiring modes: {e}")
    
    print(f"Rewiring modes: {rewiring_modes}")
    return rewiring_modes

def find_valid_scenarios(filepath, topology, scenarios, scenario_rewiring_map=None):
    """Find scenarios that exist for a specific topology."""
    print(f"Finding valid scenarios for {topology}...")
    
    valid_scenarios = []
    
    try:
        # Query for scenario-rewiring combinations that exist for this topology
        query = (
            pl.scan_csv(filepath)
            .filter(pl.col("type") == topology)
            .select(["scenario", "rewiring"])
            .unique()
            .collect()
        )
        
        # Build a map of valid scenario-rewiring combinations for this topology
        valid_combinations = {}
        for row in query.iter_rows(named=True):
            scenario = row["scenario"]
            rewiring = row["rewiring"] if row["rewiring"] else "none"
            
            # For biased and bridge, we need to include the rewiring mode
            if str(scenario).lower() in ["biased", "bridge"]:
                combined_id = f"{scenario}_{rewiring}"
                valid_combinations[combined_id] = (scenario, rewiring)
            else:
                valid_combinations[scenario] = (scenario, "none")
        
        # Check which scenarios from our list exist
        for s in scenarios:
            if "_" in s:
                # This is a combined scenario+rewiring ID
                if s in valid_combinations:
                    valid_scenarios.append(s)
            else:
                # This is a standard scenario (none, random, etc.)
                if s in valid_combinations:
                    valid_scenarios.append(s)
                # Also check combined IDs for this scenario
                for combined_id in valid_combinations:
                    if combined_id.startswith(f"{s}_"):
                        valid_scenarios.append(combined_id)
        
        # Remove duplicates
        valid_scenarios = list(set(valid_scenarios))
        
        print(f"Found {len(valid_scenarios)} valid scenarios for {topology}")
        if valid_scenarios:
            print(f"Valid scenarios: {valid_scenarios}")
        
    except Exception as e:
        print(f"Error finding valid scenarios: {e}")
        # Default to all scenarios if query fails
        valid_scenarios = scenarios
    
    return valid_scenarios

# ====================== ANALYSIS FUNCTIONS ======================

@lru_cache(maxsize=INFLECTION_CACHE_SIZE)
def find_inflection(seq_tuple):
    """
    Calculate inflection point in opinion trajectory.
    """
    seq = np.array(seq_tuple)
    
    if len(seq) < MIN_SEQUENCE_LENGTH:
        return False
    
    # Smooth trajectory and find inflection points
    try:
        smooth = gaussian_filter1d(seq, SMOOTHING_SIGMA)
        d2 = np.gradient(np.gradient(smooth))
        infls = np.where(np.diff(np.sign(d2)))[0]
    except Exception as e:
        print(f"Error in filtering: {e}")
        return False
    
    # Find first inflection point after minimum index
    inf_ind = None
    for i in infls:
        if i >= INFLECTION_MIN_IDX:
            inf_ind = i
            break
    
    # If no inflection after minimum index, but we have some inflection points, take the last one
    if inf_ind is None and len(infls) > 0:
        inf_ind = infls[-1]
    
    # No inflection points found
    if inf_ind is None:
        return False
    
    # Ensure inflection point is within reasonable range
    if inf_ind < INFLECTION_MIN_IDX or inf_ind >= len(seq) * MAX_INFLECTION_FRACTION:
        return False
    
    # Check slope direction if POSITIVE_RATES_ONLY is enabled
    if POSITIVE_RATES_ONLY:
        window_size = min(100, len(seq) - inf_ind - 1)
        if window_size <= 10 or np.polyfit(range(window_size), seq[inf_ind:inf_ind+window_size], 1)[0] <= 0:
            return False
    
    return inf_ind

@lru_cache(maxsize=CONVERGENCE_CACHE_SIZE)
def estimate_convergence_rate(trajec_tuple, loc=None, regwin=15):
    """
    Estimate convergence rate using linear regression.
    """
    trajec = np.array(trajec_tuple)
    
    if loc is None or not isinstance(loc, (int, np.integer)):
        return 0
    
    # Calculate regression
    # Ensure regression window doesn't go out of bounds
    start_idx = max(0, loc-regwin)
    end_idx = min(len(trajec)-1, loc+regwin+1)
    
    if end_idx - start_idx < 3:  # Ensure enough points for regression
        return 0
    
    x = np.arange(start_idx, end_idx)
    y = trajec[start_idx:end_idx]
    
    # Linear regression
    n = len(x)
    mx, my = np.mean(x), np.mean(y)
    ssxy = np.sum(y*x) - n*my*mx
    ssxx = np.sum(x*x) - n*mx*mx
    
    if ssxx == 0:
        return 0
    
    b1 = ssxy / ssxx
    
    # Protect against division by zero or values close to 1
    denominator = abs(trajec[loc] - 1)
    if denominator < 0.001:
        rate = b1 / 0.001
    else:
        rate = b1 / denominator
    
    return rate

def extract_run_trajectories(filepath, topology, scenario, param_value, time_range):
    """Extract trajectories for a specific combination using Polars, processing runs sequentially."""
    print(f"Extracting trajectories for {topology}/{scenario}, param={param_value}")
    
    # Set time constraints
    start_time, end_time = time_range
    
    # Handle combined scenario_rewiring format
    actual_scenario = scenario
    rewiring_filter = None
    
    if "_" in scenario:
        # Split combined ID into scenario and rewiring
        parts = scenario.split("_")
        actual_scenario = parts[0]
        rewiring_filter = parts[1]
    
    try:
        # Build the filter condition
        filter_condition = (
            (pl.col("type") == topology) & 
            (pl.col("scenario") == actual_scenario) & 
            (pl.col(PARAM_NAME) == param_value) &
            (pl.col("t") >= start_time) & 
            (pl.col("t") <= end_time)
        )
        
        # Add rewiring filter if needed
        if rewiring_filter:
            filter_condition = filter_condition & (pl.col("rewiring") == rewiring_filter)
        
        # Use Polars LazyFrame for efficient filtering
        query = pl.scan_csv(filepath).filter(filter_condition)
        
        # Get unique run IDs
        run_ids = query.select("model_run").unique().collect()["model_run"].to_list()
        
        # Decide whether to sample or process all runs
        # For processing all runs, use this:
        sampled_run_ids = run_ids  # Process all runs
        
        # Or for sampling runs, use this:
        # random.seed(RANDOM_SEED)
        # num_samples = min(MAX_SAMPLES_PER_COMBINATION, int(len(run_ids) * SAMPLE_FRACTION))
        # sampled_run_ids = random.sample(run_ids, max(1, num_samples))
        
        print(f"Processing {len(sampled_run_ids)} runs")
        
        # Extract trajectories for runs sequentially
        all_rates = []
        
        # Process runs one by one
        for i, run_id in enumerate(sampled_run_ids):
            if i % 10 == 0:  # Print progress every 10 runs
                print(f"  Processing run {i+1}/{len(sampled_run_ids)}")
            
            # Process single run
            rate = process_single_run(filepath, topology, scenario, param_value, run_id, time_range)
            
            # Add rate to results if it meets the criteria
            if not POSITIVE_RATES_ONLY or rate > 0:
                all_rates.append(rate * 1000)  # Scale for visualization
            
            # Optional: Garbage collect periodically to manage memory
            if i % 50 == 0 and i > 0:
                gc.collect()
        
        print(f"Found {len(all_rates)} valid rates")
        return all_rates
        
    except Exception as e:
        print(f"Error extracting trajectories: {e}")
        traceback.print_exc()
        return []

def process_single_run(filepath, topology, scenario, param_value, run_id, time_range):
    """Process a single run to calculate convergence rate."""
    try:
        # Handle combined scenario_rewiring format
        actual_scenario = scenario
        rewiring_filter = None
        
        if "_" in scenario:
            # Split combined ID into scenario and rewiring
            parts = scenario.split("_")
            actual_scenario = parts[0]
            rewiring_filter = parts[1]
        
        # Build the filter condition
        filter_condition = (
            (pl.col("type") == topology) & 
            (pl.col("scenario") == actual_scenario) & 
            (pl.col(PARAM_NAME) == param_value) &
            (pl.col("model_run") == run_id) &
            (pl.col("t") >= time_range[0]) & 
            (pl.col("t") <= time_range[1])
        )
        
        # Add rewiring filter if needed
        if rewiring_filter:
            filter_condition = filter_condition & (pl.col("rewiring") == rewiring_filter)
            
        # Extract data for this run
        run_data = (
            pl.scan_csv(filepath)
            .filter(filter_condition)
            .select(["t", "avg_state"])
            .sort("t")
            .collect()
        )
        
        # Check if we have enough data points
        if len(run_data) < MIN_SEQUENCE_LENGTH:
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

def save_individual_results(results, topology, param_name, time_range, output_dir=OUTPUT_DIR):
    """Save processed results for a single topology to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with key parameters
    start_time, end_time = time_range
    positive_suffix = "_positive" if POSITIVE_RATES_ONLY else "_all"
    
    filename = f"convergence_rates_{topology}_{param_name}_{start_time}_{end_time}{positive_suffix}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Save to pickle
    with open(filepath, 'wb') as f:
        pickle.dump({
            'results': results,
            'topology': topology,
            'param_name': param_name,
            'time_range': time_range,
            'positive_only': POSITIVE_RATES_ONLY,
            'date_processed': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)
    
    print(f"Individual results saved to {filepath}")

def save_all_results(all_results, param_name, time_range, output_dir=OUTPUT_DIR):
    """Save all processed results to a single file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with key parameters
    start_time, end_time = time_range
    positive_suffix = "_positive" if POSITIVE_RATES_ONLY else "_all"
    
    filename = f"convergence_rates_ALL_{param_name}_{start_time}_{end_time}{positive_suffix}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Save to pickle
    with open(filepath, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'param_name': param_name,
            'time_range': time_range,
            'positive_only': POSITIVE_RATES_ONLY,
            'date_processed': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)
    
    print(f"All results saved to {filepath}")

def main():
    """Main execution function."""
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Limit CPU usage
    limit_cpu_cores(int(0.6 * os.cpu_count()))
    
    # Get input file path
    input_file = "../Output/param_sweep_individual_N_789_polarisingNode_f_0.0_1.0_10_2025-03-13.csv" # get_data_file()  
    
    try:
        print(f"Processing file: {input_file}")
        print(f"Analysis mode: {'POSITIVE_RATES_ONLY' if POSITIVE_RATES_ONLY else 'ALL_RATES'}")
        print_memory_usage("at start")
        
        # Get file schema
        headers, dtypes = get_schema_info(input_file)
        if not headers:
            print("Error: Could not read schema from file")
            return
        
        # Get topologies and scenarios from the data
        available_topologies, available_scenarios = find_topologies_and_scenarios(input_file)
        
        # Apply filters based on user selection
        if SELECTED_TOPOLOGIES:
            topologies = [t for t in SELECTED_TOPOLOGIES if t in available_topologies]
            print(f"Using selected topologies: {topologies}")
        else:
            topologies = available_topologies
            print(f"Using all available topologies: {topologies}")
        
        # Get parameter values
        param_values = get_parameter_values(input_file, PARAM_NAME)
        
        # Get rewiring modes - this needs to identify biased_same, biased_diff, etc.
        rewiring_modes = get_rewiring_modes(input_file, available_scenarios)
        
        # Find all unique scenario-rewiring combinations
        unique_combinations = set()
        # First fetch all unique scenario-rewiring pairs from data
        try:
            query = (
                pl.scan_csv(input_file)
                .select(["scenario", "rewiring"])
                .unique()
                .collect()
            )
            
            # Process to create combined scenario_rewiring for biased and bridge
            for row in query.iter_rows(named=True):
                scenario = row["scenario"]
                rewiring = row["rewiring"]
                
                if str(scenario).lower() in ["biased", "bridge"] and rewiring:
                    combined_id = f"{scenario}_{rewiring}"
                    unique_combinations.add(combined_id)
                else:
                    unique_combinations.add(scenario)
        except Exception as e:
            print(f"Error getting unique combinations: {e}")
            # Fall back to available scenarios
            unique_combinations = set(available_scenarios)
        
        # Convert to list and filter based on user selection
        all_scenario_combinations = list(unique_combinations)
        if SELECTED_SCENARIOS:
            # Include both direct matches and combined IDs starting with selected scenarios
            scenarios = []
            for s in SELECTED_SCENARIOS:
                if s in all_scenario_combinations:
                    scenarios.append(s)
                for combined in all_scenario_combinations:
                    if combined.startswith(f"{s}_"):
                        scenarios.append(combined)
            print(f"Using selected scenarios/combinations: {scenarios}")
        else:
            scenarios = all_scenario_combinations
            print(f"Using all scenario combinations: {scenarios}")
            
        # Dictionary to accumulate all results
        all_results = {}
        
        # Process each selected topology
        for topology in topologies:
            # Process the topology
            results = process_topology(
                input_file, topology, scenarios,
                param_values, rewiring_modes
            )
            
            # Save individual results
            if results:
                save_individual_results(results, topology, PARAM_NAME, TIME_RANGE)
                
                # Add to all results
                all_results[topology] = results
            
            # Clear memory
            results = {}
            gc.collect()
            print_memory_usage(f"after processing topology {topology}")
        
        # Save all results to a single file
        if all_results:
            save_all_results(all_results, PARAM_NAME, TIME_RANGE)
        
        end_time = time.time()
        print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()