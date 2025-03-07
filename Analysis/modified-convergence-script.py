import sys
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from datetime import date
import gc
import time

# Global settings
FONT_SIZE = 14
cm = 1/2.54
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

# Define friendly names mapping
friendly_names = {
    'None_none': 'static',
    'random_none': 'random',
    'biased_same': 'local (similar)',
    'biased_diff': 'local (opposite)',
    'bridge_same': 'bridge (similar)',
    'bridge_diff': 'bridge (opposite)',
    'wtf_none': 'wtf',
    'node2vec_none': 'node2vec'
}

def setup_plotting_style():
    """Set consistent style elements for all plots"""
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 1.5,
        'lines.linewidth': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (11.4*cm, 7*cm), 
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })
    
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        "axes.grid": True,
        "grid.color": 'black',
        'grid.linestyle': 'solid', 
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.5,
        "axes.spines.bottom": True,
        "grid.alpha": 0.4,
        "xtick.bottom": True,
        "ytick.left": True
    })

# Using the convergence rate calculation from convergence_rate_plots_average.py
def find_inflection(seq, min_idx=5000, sigma=300):
    """Calculate inflection point in trajectory"""
    smooth = gaussian_filter1d(seq, sigma)
    d2 = np.gradient(np.gradient(smooth))
    infls = np.where(np.diff(np.sign(d2)))[0]
    
    # Find first inflection point after min_idx
    inf_ind = None
    for i in infls:
        if i >= min_idx:
            inf_ind = i
            break
    
    # If no inflection after min_idx but we have some inflection points, take the last one
    if inf_ind is None and len(infls) > 0:
        inf_ind = infls[-1]
    
    # No inflection points found
    if inf_ind is None:
        return False, smooth
    
    # Ensure inflection point is within reasonable range
    if inf_ind < min_idx or inf_ind >= len(seq) * 0.9:  # Avoid points too close to the end
        return False, smooth
    
    return inf_ind, smooth

def estimate_convergence_rate(trajec, loc=None, regwin=15):
    """Estimate convergence rate around specified location"""
    if loc is None or not isinstance(loc, (int, np.integer)):
        return 0
    
    x = np.arange(len(trajec))
    y = trajec
    
    if loc is not None:
        # Ensure regression window doesn't go out of bounds
        start_idx = max(0, loc-regwin)
        end_idx = min(len(trajec)-1, loc+regwin+1)
        x = x[start_idx:end_idx]
        y = trajec[start_idx:end_idx]
    
    # Ensure we have enough points for regression
    if len(x) < 3:
        return 0
    
    # Linear regression
    n = np.size(x) 
    mx, my = np.mean(x), np.mean(y) 
    ssxy = np.sum(y*x) - n*my*mx 
    ssxx = np.sum(x*x) - n*mx*mx 
    
    # Avoid division by zero
    if ssxx == 0:
        return 0
        
    b1 = ssxy / ssxx 
    b0 = my - b1*mx 
    
    # Protect against division by zero or values close to 1
    if abs(trajec[loc] - 1) < 0.001:
        rate = -b1/0.001
    else:
        rate = -b1/(trajec[loc]-1)
    
    return rate

def get_scenario_combinations(filepath, chunksize=10000):
    """
    Get unique scenario, rewiring, type combinations from the data
    without loading the entire file
    """
    unique_combinations = set()
    
    # Count total lines for progress reporting
    total_lines = sum(1 for _ in open(filepath, 'r'))
    chunk_count = total_lines // chunksize
    
    print(f"Scanning file for unique scenario combinations ({total_lines:,} lines)")
    
    # Process in chunks to identify unique combinations
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize, usecols=['scenario', 'rewiring', 'type'])):
        # Fill NA values in categorical columns
        chunk['rewiring'] = chunk['rewiring'].fillna('none')
        chunk['scenario'] = chunk['scenario'].fillna('none')
        
        # Create combined scenario names to match expected classification
        for _, row in chunk.iterrows():
            scenario = row['scenario']
            rewiring = row['rewiring']
            topology = row['type']
            
            # Specifically look for all expected combinations
            if scenario == 'None' and rewiring == 'none':
                scenario_key = ('None', 'none', topology)  # static
            elif scenario == 'random' and rewiring == 'none':
                scenario_key = ('random', 'none', topology)  # random
            elif scenario == 'biased' and rewiring == 'same':
                scenario_key = ('biased', 'same', topology)  # local (similar)
            elif scenario == 'biased' and rewiring == 'diff':
                scenario_key = ('biased', 'diff', topology)  # local (opposite)
            elif scenario == 'bridge' and rewiring == 'same':
                scenario_key = ('bridge', 'same', topology)  # bridge (similar)
            elif scenario == 'bridge' and rewiring == 'diff':
                scenario_key = ('bridge', 'diff', topology)  # bridge (opposite)
            elif scenario == 'wtf' and rewiring == 'none':
                scenario_key = ('wtf', 'none', topology)  # wtf
            elif scenario == 'node2vec' and rewiring == 'none':
                scenario_key = ('node2vec', 'none', topology)  # node2vec
            else:
                scenario_key = (scenario, rewiring, topology)
                
            unique_combinations.add(scenario_key)
        
        # Progress update
        if (i+1) % 10 == 0 or (i+1) == chunk_count:
            print(f"  Processed chunk {i+1}/{chunk_count} ({len(unique_combinations)} unique combinations found)")
    
    # Convert to list of tuples for easier handling
    return sorted(list(unique_combinations))

def process_trajectories_by_combination(filepath, combination, chunksize=10000, use_sampling=False, max_samples=100):
    """
    Process trajectories for a specific scenario combination
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    combination : tuple
        (scenario, rewiring, type) combination to process
    chunksize : int
        Size of chunks to read from file
    use_sampling : bool
        Whether to sample trajectories (True) or use all (False)
    max_samples : int
        Maximum number of trajectories to sample if use_sampling is True
    
    Returns:
    --------
    list
        List of dictionaries containing rate calculations
    """
    scenario, rewiring, topology = combination
    scenario_combined = f"{scenario}_{rewiring}"
    friendly_scenario = friendly_names.get(scenario_combined, scenario_combined)
    
    rates_list = []
    processed_runs = set()
    target_runs = set()
    
    print(f"\nProcessing {scenario}_{rewiring}_{topology}:")
    
    # First pass: identify all model runs for this combination
    total_runs = 0
    
    # Read only model_run column first to identify all runs
    for chunk in pd.read_csv(filepath, chunksize=chunksize, 
                           usecols=['scenario', 'rewiring', 'type', 'model_run']):
        # Fill NA values
        chunk['rewiring'] = chunk['rewiring'].fillna('none')
        chunk['scenario'] = chunk['scenario'].fillna('none')
        
        # Filter for the target combination
        mask = ((chunk['scenario'] == scenario) & 
                (chunk['rewiring'] == rewiring) & 
                (chunk['type'] == topology))
        
        if not mask.any():
            continue
            
        # Get unique model runs in this chunk for this combination
        chunk_runs = set(chunk.loc[mask, 'model_run'].unique())
        total_runs += len(chunk_runs)
        
        # Add to our potential runs
        target_runs.update(chunk_runs)
    
    print(f"  Found {len(target_runs)} model runs for this combination")
    
    # Determine which runs to process
    runs_to_process = set()
    if use_sampling and len(target_runs) > max_samples:
        # If sampling is enabled and we have more runs than max_samples
        runs_to_process = set(np.random.choice(list(target_runs), size=max_samples, replace=False))
        print(f"  Sampling enabled: Processing {len(runs_to_process)} of {len(target_runs)} runs")
    else:
        # Use all runs (no sampling)
        runs_to_process = target_runs
        print(f"  Processing all {len(runs_to_process)} runs")
    
    # Second pass: process all selected model runs
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Fill NA values
        chunk['rewiring'] = chunk['rewiring'].fillna('none')
        chunk['scenario'] = chunk['scenario'].fillna('none')
        
        # Filter for the target combination and selected runs
        mask = ((chunk['scenario'] == scenario) & 
                (chunk['rewiring'] == rewiring) & 
                (chunk['type'] == topology) &
                (chunk['model_run'].isin(runs_to_process)))
        
        if not mask.any():
            continue
        
        # Group by model_run
        for run, group in chunk[mask].groupby('model_run'):
            # Skip if we've already processed this run
            if run in processed_runs:
                continue
                
            # Skip if we don't have enough data points yet
            if len(group) < 1200:
                continue
                
            # Check if we have the complete trajectory
            has_full_trajectory = group['t'].max() >= 20000
                
            # If we have a full trajectory, process it
            if has_full_trajectory:
                # Sort by timestep
                group = group.sort_values('t')
                
                # Get the trajectory
                trajectory = group['avg_state'].values
                
                # Find inflection point using the method from convergence_rate_plots_average.py
                inflection_x, smoothed = find_inflection(trajectory)
                
                if inflection_x:
                    try:
                        # Use the estimate_convergence_rate method from convergence_rate_plots_average.py
                        rate = estimate_convergence_rate(smoothed, loc=inflection_x)
                        rates_list.append({
                            'scenario': friendly_scenario,
                            'topology': topology,
                            'model_run': run,
                            'rate': rate * 1000,  # Scale rate as in original code
                            'original_scenario': scenario,
                            'original_rewiring': rewiring
                        })
                        
                        # Mark as processed
                        processed_runs.add(run)
                        
                        # Progress update
                        if len(processed_runs) % 10 == 0:
                            print(f"  Processed {len(processed_runs)}/{len(runs_to_process)} runs")
                    except Exception as e:
                        print(f"  Error calculating rate for run {run}: {str(e)}")
    
    print(f"  Completed processing {len(processed_runs)}/{len(runs_to_process)} runs")
    print(f"  Successfully calculated rates for {len(rates_list)} runs")
    
    # Clean up memory
    gc.collect()
    
    return rates_list

def plot_convergence_rates(rates_df, output_path):
    """Create convergence rate comparison plot with topologies averaged
    
    This creates a plot that shows one data point per topology/scenario combination,
    with each data point representing the average per topology and scenario.
    The red line shows the average of all topologies for each rewiring strategy.
    """
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(11, 8))
    
    # First, calculate the average rate for each topology and scenario combination
    topology_scenario_avg = rates_df.groupby(['topology', 'scenario'])['rate'].mean().reset_index()
    
    # Now calculate the overall average per scenario across all topologies
    scenario_overall_avg = topology_scenario_avg.groupby('scenario')['rate'].mean()
    
    # Sort scenarios by their overall average rate
    scenario_order = scenario_overall_avg.sort_values(ascending=False).index.tolist()
    
    # Create a categorical column for ordered scenarios
    topology_scenario_avg['scenario'] = pd.Categorical(
        topology_scenario_avg['scenario'], 
        categories=scenario_order, 
        ordered=True
    )
    
    # Define traditional statistical physics markers and their labels
    topology_markers = {
        'DPAH': 'x',      # cross
        'cl': '+',        # plus
        'Twitter': '*',   # asterisk
        'FB': '.'         # point
    }
    
    # Create x-coordinates for scenarios
    x_coords = np.arange(len(scenario_order))
    
    # Create main scatter plot with topology markers - now using the aggregated data
    for topology, marker in topology_markers.items():
        mask = topology_scenario_avg['topology'] == topology
        scenario_data = topology_scenario_avg[mask]
        
        # Map scenarios to x-coordinates
        scenario_x = [x_coords[list(scenario_order).index(s)] for s in scenario_data['scenario']]
        
        plt.scatter(
            scenario_x,
            scenario_data['rate'],
            marker=marker,
            s=100,  # Marker size
            c='black',  # All markers in black
            label=topology,
            alpha=0.7
        )
    
    # Add overall average values as horizontal lines
    for i, scenario in enumerate(scenario_order):
        avg_rate = scenario_overall_avg[scenario]
        plt.hlines(y=avg_rate, xmin=i-0.2, xmax=i+0.2, 
                  colors='red', linestyles='solid', alpha=0.5)
    
    # Customize the plot
    plt.title('Convergence Rates Comparison', pad=20, fontsize=FONT_SIZE)
    plt.xlabel('Rewiring Strategy', labelpad=10, fontsize=FONT_SIZE)
    plt.ylabel('Convergence Rate (×10³)', labelpad=10, fontsize=FONT_SIZE)
    
    # Set x-axis ticks and labels
    plt.xticks(x_coords, scenario_order, 
               rotation=45, 
               horizontalalignment='right',
               rotation_mode='anchor',
               fontsize=FONT_SIZE)
    
    # Create legend for topologies (markers)
    legend_elements = [plt.Line2D([0], [0], marker=marker, color='black', 
                                 label=topology, markersize=10, linestyle='None')
                      for topology, marker in topology_markers.items()]
    
    # Keep the legend label as "Median" to match your example
    legend_elements.append(plt.Line2D([0], [0], color='red', alpha=0.5,
                                     label='Median', linestyle='solid'))
    
    # Place legend
    plt.legend(handles=legend_elements, title="Network Topology", 
              loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    plt.yticks(fontsize=FONT_SIZE)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight', 
                format='pdf',
                metadata={'Creator': "Modified Script", 'Producer': None})
    plt.show()
    plt.close()

def print_summary_statistics(rates_df):
    """Print summary statistics for convergence rates by scenario and topology"""
    # Calculate mean and other stats per scenario
    print("\nConvergence Rate Summary Statistics by Scenario:")
    scenario_stats = rates_df.groupby(['scenario'])['rate'].agg([
        ('Mean', np.mean),
        ('Median', np.median),
        ('Std', np.std),
        ('Min', np.min),
        ('Max', np.max),
        ('Count', 'count')
    ]).round(3)
    print(scenario_stats)
    
    # Calculate mean and other stats per scenario and topology
    print("\nConvergence Rate Summary Statistics by Scenario and Topology:")
    detailed_stats = rates_df.groupby(['scenario', 'topology'])['rate'].agg([
        ('Mean', np.mean),
        ('Median', np.median),
        ('Std', np.std),
        ('Count', 'count')
    ]).round(3)
    print(detailed_stats)
    
    # Calculate topology-level aggregations
    print("\nTopology-Level Aggregations:")
    topology_stats = rates_df.groupby(['topology'])['rate'].agg([
        ('Mean', np.mean),
        ('Median', np.median),
        ('Std', np.std),
        ('Count', 'count')
    ]).round(3)
    print(topology_stats)
    
    return scenario_stats, detailed_stats, topology_stats

def save_statistics(stats, detailed_stats, topology_stats, output_base):
    """Save summary statistics to CSV files"""
    stats.to_csv(f"{output_base}_summary_by_scenario.csv")
    detailed_stats.to_csv(f"{output_base}_summary_by_scenario_topology.csv")
    topology_stats.to_csv(f"{output_base}_summary_by_topology.csv")
    print(f"Statistics saved to {output_base}_summary_*.csv")

def save_rates_df(rates_df, output_base):
    """Save the calculated rates to a CSV file for future use"""
    rates_file = f"{output_base}_rates.csv"
    rates_df.to_csv(rates_file, index=False)
    print(f"Rates data saved to {rates_file}")
    return rates_file

def load_saved_rates(rates_file):
    """Load previously calculated rates from a CSV file"""
    if os.path.exists(rates_file):
        print(f"Loading previously calculated rates from {rates_file}")
        return pd.read_csv(rates_file)
    return None

def main():
    start_time = time.time()
    
    # Get list of relevant output files
    file_list = [f for f in os.listdir("../Output") if f.endswith(".csv") and "default_run_individual" in f]

    if not file_list:
        print("No suitable individual data files found in the Output directory.")
        exit()

    # Print file list with indices
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")

    # Get user input for file selection
    file_index = int(input("Enter the index of the file you want to analyze: "))
    
    # Get user input for chunk size
    chunk_size = int(input("Chunk size for processing (default: 50000): ") or "50000")
    
    # Get user input for calculation method
    use_sampling = input("Enable sampling to reduce memory usage? (y/n, default: n): ").lower().startswith('y')
    if use_sampling:
        max_samples = int(input("Maximum samples per combination (default: 50): ") or "50")
    else:
        max_samples = None  # Not used when sampling is disabled
    
    # Load the selected file path
    data_path = os.path.join("../Output", file_list[file_index])
    print(f"Processing file: {data_path}")
    
    # Generate output filename base
    N = file_list[file_index].split("_")[5]  # Extract N from filename
    method_suffix = "_individual"
    if use_sampling:
        method_suffix += "_sampled"
    
    output_base = f"../Output/convergence{method_suffix}_N{N}_{date.today()}"
    rates_file = f"{output_base}_rates.csv"
    
    # Check if we have previously calculated rates
    saved_rates_df = load_saved_rates(rates_file)
    
    if saved_rates_df is not None:
        # Use previously calculated rates
        rates_df = saved_rates_df
        print(f"Using {len(rates_df)} previously calculated rates")
    else:
        # Calculate rates using the selected method
        # Find all unique scenario combinations in the file
        combinations = get_scenario_combinations(data_path, chunksize=chunk_size)
        print(f"Found {len(combinations)} unique scenario combinations")
        
        # Process each combination separately
        all_rates = []
        for i, combination in enumerate(combinations):
            print(f"Combination {i+1}/{len(combinations)}: {combination}")
            
            # Process combination with or without sampling based on user preference
            combination_rates = process_trajectories_by_combination(
                data_path, 
                combination, 
                chunksize=chunk_size,
                use_sampling=use_sampling,
                max_samples=max_samples if use_sampling else None
            )
            all_rates.extend(combination_rates)
            
            # Intermediate progress update
            print(f"Processed {i+1}/{len(combinations)} combinations. Total rates: {len(all_rates)}")
            
            # Save intermediate results every few combinations
            if (i+1) % 3 == 0 or (i+1) == len(combinations):
                if all_rates:  # Only save if we have results
                    interim_df = pd.DataFrame(all_rates)
                    interim_file = f"{output_base}_interim_{i+1}.csv"
                    interim_df.to_csv(interim_file, index=False)
                    print(f"Saved interim results to {interim_file}")
        
        # Convert results to DataFrame
        if all_rates:
            rates_df = pd.DataFrame(all_rates)
        else:
            print("No valid rates were calculated. Check your data.")
            return None
        
        # Save the calculated rates for future use
        if rates_df is not None and not rates_df.empty:
            save_rates_df(rates_df, output_base)
    
    if len(rates_df) == 0:
        print("Warning: No valid convergence rates calculated")
        return
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create output directory if it doesn't exist
    os.makedirs("../Figs/Convergence", exist_ok=True)
    
    # Generate output filename
    output_path = f"../Figs/Convergence/convergence_rates{method_suffix}_N{N}_{date.today()}.pdf"
    
    # Create and save the plot
    plot_convergence_rates(rates_df, output_path)
    print(f"Convergence rate plot saved to: {output_path}")
    
    # Calculate and print summary statistics
    scenario_stats, detailed_stats, topology_stats = print_summary_statistics(rates_df)
    
    # Save statistics to files
    save_statistics(scenario_stats, detailed_stats, topology_stats, output_base)
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")
    
    return rates_df

if __name__ == "__main__":
    df_rates = main()