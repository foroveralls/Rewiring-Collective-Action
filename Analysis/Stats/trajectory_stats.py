"""
Trajectory Analysis Script - Analyzes algorithm performance from individual trajectory data.
Processes data in batches to handle large files and produces properly stacked metrics.
"""

import pandas as pd
import numpy as np
import os
from datetime import date
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Constants
FRIENDLY_NAMES = {
    'none_none': 'static', 'random_none': 'random', 'biased_same': 'local (similar)',
    'biased_diff': 'local (opposite)', 'bridge_same': 'bridge (similar)', 
    'bridge_diff': 'bridge (opposite)', 'wtf_none': 'wtf', 'node2vec_none': 'node2vec'
}
DIRECTED_NETWORKS = ['DPAH', 'Twitter']
UNDIRECTED_NETWORKS = ['cl', 'FB']
BATCH_SIZE = 800000


def get_friendly_name(scenario_grouped):
    """Convert internal scenario name to human-readable format."""
    parts = scenario_grouped.split('_')
    key = f"{parts[0]}_{parts[1]}"
    return FRIENDLY_NAMES.get(key, scenario_grouped)


def find_inflection(seq):
    """Find inflection point in trajectory (proxy for convergence)."""
    if len(seq) < 1200:
        return False
    
    try:
        smooth = gaussian_filter1d(seq, 600)
        d2 = np.gradient(np.gradient(smooth))
        infls = np.where(np.diff(np.sign(d2)))[0]
        
        inf_min = 5000
        if not any(i >= inf_min for i in infls):
            return False
            
        inf_ind = next((i for i in infls if i >= inf_min), False)
        return inf_ind if inf_ind > inf_min and inf_ind < 20000 else False
    except:
        return False


def estimate_convergence_rate(trajec, loc, regwin=10):
    """Estimate convergence rate around specified location."""
    x = np.arange(len(trajec) - 1)
    y = trajec 
    
    if loc is not None:
        x = x[loc-regwin: loc+regwin+1]
        y = trajec[loc-regwin: loc+regwin+1]
    
    y, x = np.asarray([y, x])
    
    # Linear regression
    n = np.size(x) 
    mx, my = np.mean(x), np.mean(y) 
    ssxy = np.sum(y*x) - n*my*mx 
    ssxx = np.sum(x*x) - n*mx*mx 
    b1 = ssxy / ssxx 
    
    rate = -b1/(trajec[loc]-1)
    return rate


def analyze_trajectory_data(file_path, t_max=40000, batch_size=BATCH_SIZE):
    """Main function to analyze trajectory data in batches."""
    print(f"Processing file: {file_path}")
    
    # Process file in batches to handle large datasets
    all_metrics = []
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=batch_size)):
        print(f"Processing batch {i+1}...")
        
        # Prepare batch
        batch = chunk.copy()
        batch['rewiring'] = batch['rewiring'].fillna('none')
        batch['scenario'] = batch['scenario'].fillna('none')
        
        # Extract metrics from each trajectory
        batch_results = []
        for name, group in batch.groupby(['scenario', 'rewiring', 'type', 'model_run']):
            scenario, rewiring, topology, run = name
            
            # Filter data and skip if invalid
            group = group[group['t'] <= t_max]
            if group.empty or len(group) < 1200:
                continue
            
            # Get trajectory data
            trajectory = group['avg_state'].values
            polarization = group['std_states'].values
            
            # Calculate friendly name and network class
            friendly_name = get_friendly_name(f"{scenario}_{rewiring}")
            network_class = 'directed' if topology in DIRECTED_NETWORKS else 'undirected'
            
            # Extract algorithm characteristics
            is_similar = '(similar)' in friendly_name
            is_opposite = '(opposite)' in friendly_name
            algorithm_base = friendly_name.split(' (')[0] if is_similar or is_opposite else friendly_name
            rewiring_type = 'similar' if is_similar else 'opposite' if is_opposite else 'none'
            
            # Calculate convergence metrics
            convergence_rate = np.nan
            inflection_x = find_inflection(trajectory)
            if inflection_x:
                try:
                    convergence_rate = estimate_convergence_rate(trajectory, inflection_x) * 1000
                except:
                    pass
            
            # Calculate final metrics (from last 5000 timesteps)
            window_start = t_max - 5000
            final_window = group[group['t'] >= window_start]
            final_cooperativity = final_window['avg_state'].mean() if not final_window.empty else trajectory[-1]
            final_polarization = final_window['std_states'].mean() if not final_window.empty else polarization[-1]
            
            # Calculate time to cooperative majority
            coop_majority_idx = np.argmax(trajectory > 0.5) if any(trajectory > 0.5) else -1
            time_to_majority = group.iloc[coop_majority_idx]['t'] if coop_majority_idx >= 0 else np.nan
            
            # Store trajectory metrics
            batch_results.append({
                'scenario': scenario, 
                'rewiring': rewiring, 
                'topology': topology, 
                'model_run': run, 
                'friendly_name': friendly_name, 
                'network_class': network_class,
                'algorithm_base': algorithm_base,
                'rewiring_type': rewiring_type,
                'convergence_rate': convergence_rate, 
                'final_cooperativity': final_cooperativity,
                'final_polarization': final_polarization, 
                'time_to_majority': time_to_majority
            })
        
        # Add batch results if not empty
        if batch_results:
            all_metrics.append(pd.DataFrame(batch_results))
    
    # Combine results from all batches
    if not all_metrics:
        print("No valid metrics calculated.")
        return None
        
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # Calculate comparison metrics against baselines
    print("Calculating comparative metrics...")
    metrics_with_comparisons = calculate_baseline_comparisons(metrics_df)
    
    # Calculate summaries
    print("Generating summary statistics...")
    summaries = calculate_summaries(metrics_with_comparisons)
    
    return summaries


def calculate_baseline_comparisons(metrics_df):
    """Calculate metrics relative to static and random baselines."""
    results = []
    
    # For each topology, compare algorithms to baselines
    for topology in metrics_df['topology'].unique():
        # Get baseline data
        static_data = metrics_df[(metrics_df['topology'] == topology) & 
                               (metrics_df['friendly_name'] == 'static')]
        random_data = metrics_df[(metrics_df['topology'] == topology) & 
                               (metrics_df['friendly_name'] == 'random')]
        
        # Skip if missing baselines
        if static_data.empty or random_data.empty:
            continue
            
        # Calculate baseline averages
        baselines = {
            'static': {
                'rate': static_data['convergence_rate'].mean(),
                'coop': static_data['final_cooperativity'].mean(),
                'polar': static_data['final_polarization'].mean(),
                'time': static_data['time_to_majority'].mean()
            },
            'random': {
                'rate': random_data['convergence_rate'].mean(),
                'coop': random_data['final_cooperativity'].mean(),
                'polar': random_data['final_polarization'].mean(),
                'time': random_data['time_to_majority'].mean()
            }
        }
        
        # Compare each algorithm against baselines
        topo_data = metrics_df[metrics_df['topology'] == topology]
        for _, row in topo_data.iterrows():
            # Skip baselines
            if row['friendly_name'] in ['static', 'random']:
                continue
                
            # Calculate relative metrics
            comparisons = {}
            for baseline in ['static', 'random']:
                for metric, value_key in [
                    ('rate', 'convergence_rate'), 
                    ('coop', 'final_cooperativity'),
                    ('polar', 'final_polarization'),
                    ('time', 'time_to_majority')
                ]:
                    # Calculate relative value
                    baseline_val = baselines[baseline][metric]
                    if not np.isnan(baseline_val) and baseline_val != 0:
                        rel_val = row[value_key] / baseline_val
                        
                        # Determine if better (for rate/coop higher is better, for polar/time lower is better)
                        is_better = rel_val > 1 if metric in ['rate', 'coop'] else rel_val < 1
                        
                        comparisons[f"rel_vs_{baseline}_{metric}"] = rel_val
                        comparisons[f"better_than_{baseline}_{metric}"] = is_better
                    else:
                        comparisons[f"rel_vs_{baseline}_{metric}"] = np.nan
                        comparisons[f"better_than_{baseline}_{metric}"] = np.nan
            
            # Add to results
            results.append({**row.to_dict(), **comparisons})
    
    return pd.DataFrame(results)


def calculate_summaries(comparative_metrics):
    """Calculate summary statistics and aggregate results."""
    # Filter non-baseline algorithms
    non_baseline = comparative_metrics[~comparative_metrics['friendly_name'].isin(['static', 'random'])]
    
    # 1. Individual run statistics
    run_stats = {
        'metric_type': 'individual_run_statistics',
        'total_runs': len(non_baseline),
        'vs_static_rate_count': non_baseline['better_than_static_rate'].sum(),
        'vs_static_rate_pct': non_baseline['better_than_static_rate'].mean() * 100,
        'vs_static_coop_count': non_baseline['better_than_static_coop'].sum(),
        'vs_static_coop_pct': non_baseline['better_than_static_coop'].mean() * 100,
        'vs_random_rate_count': non_baseline['better_than_random_rate'].sum(),
        'vs_random_rate_pct': non_baseline['better_than_random_rate'].mean() * 100,
        'vs_random_coop_count': non_baseline['better_than_random_coop'].sum(),
        'vs_random_coop_pct': non_baseline['better_than_random_coop'].mean() * 100
    }
    
    # 2. Algorithm type statistics (average of runs by algorithm type)
    # Use a more direct approach to avoid multi-index issues
    algo_stats = pd.DataFrame()
    algo_stats['friendly_name'] = non_baseline['friendly_name'].unique()
    
    # Calculate metrics for each algorithm type
    for algo in algo_stats['friendly_name']:
        algo_data = non_baseline[non_baseline['friendly_name'] == algo]
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'better_than_static_rate_mean'] = algo_data['better_than_static_rate'].mean()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'better_than_static_coop_mean'] = algo_data['better_than_static_coop'].mean()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'better_than_random_rate_mean'] = algo_data['better_than_random_rate'].mean()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'better_than_random_coop_mean'] = algo_data['better_than_random_coop'].mean()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'convergence_rate_mean'] = algo_data['convergence_rate'].mean()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'convergence_rate_median'] = algo_data['convergence_rate'].median()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'final_cooperativity_mean'] = algo_data['final_cooperativity'].mean()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'final_cooperativity_median'] = algo_data['final_cooperativity'].median()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'final_polarization_mean'] = algo_data['final_polarization'].mean()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'final_polarization_median'] = algo_data['final_polarization'].median()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'time_to_majority_mean'] = algo_data['time_to_majority'].mean()
        algo_stats.loc[algo_stats['friendly_name'] == algo, 'time_to_majority_median'] = algo_data['time_to_majority'].median()
    
    # Add metric_type column
    algo_stats['metric_type'] = 'algorithm_summary'
    
    # Count algorithm types better than baselines (>50% of runs)
    algo_counts = {
        'metric_type': 'algorithm_type_statistics',
        'total_algorithm_types': len(algo_stats),
        'algo_better_static_rate': sum(algo_stats['better_than_static_rate_mean'] > 0.5),
        'algo_better_static_rate_pct': sum(algo_stats['better_than_static_rate_mean'] > 0.5) / len(algo_stats) * 100,
        'algo_better_static_coop': sum(algo_stats['better_than_static_coop_mean'] > 0.5),
        'algo_better_static_coop_pct': sum(algo_stats['better_than_static_coop_mean'] > 0.5) / len(algo_stats) * 100,
        'algo_better_random_rate': sum(algo_stats['better_than_random_rate_mean'] > 0.5),
        'algo_better_random_rate_pct': sum(algo_stats['better_than_random_rate_mean'] > 0.5) / len(algo_stats) * 100,
        'algo_better_random_coop': sum(algo_stats['better_than_random_coop_mean'] > 0.5),
        'algo_better_random_coop_pct': sum(algo_stats['better_than_random_coop_mean'] > 0.5) / len(algo_stats) * 100
    }
    
    # 3. Network class statistics - ENHANCED with more metrics
    # Use a simpler approach to avoid multi-index issues
    network_stats = pd.DataFrame()
    network_stats['network_class'] = non_baseline['network_class'].unique()
    
    # Calculate metrics for each network class
    for network_class in network_stats['network_class']:
        class_data = non_baseline[non_baseline['network_class'] == network_class]
        network_stats.loc[network_stats['network_class'] == network_class, 'better_than_static_rate_mean'] = class_data['better_than_static_rate'].mean()
        network_stats.loc[network_stats['network_class'] == network_class, 'better_than_static_coop_mean'] = class_data['better_than_static_coop'].mean()
        network_stats.loc[network_stats['network_class'] == network_class, 'better_than_random_rate_mean'] = class_data['better_than_random_rate'].mean()
        network_stats.loc[network_stats['network_class'] == network_class, 'better_than_random_coop_mean'] = class_data['better_than_random_coop'].mean()
        network_stats.loc[network_stats['network_class'] == network_class, 'convergence_rate_mean'] = class_data['convergence_rate'].mean()
        network_stats.loc[network_stats['network_class'] == network_class, 'convergence_rate_median'] = class_data['convergence_rate'].median()
        network_stats.loc[network_stats['network_class'] == network_class, 'final_cooperativity_mean'] = class_data['final_cooperativity'].mean()
        network_stats.loc[network_stats['network_class'] == network_class, 'final_cooperativity_median'] = class_data['final_cooperativity'].median()
        network_stats.loc[network_stats['network_class'] == network_class, 'final_polarization_mean'] = class_data['final_polarization'].mean()
        network_stats.loc[network_stats['network_class'] == network_class, 'final_polarization_median'] = class_data['final_polarization'].median()
        network_stats.loc[network_stats['network_class'] == network_class, 'time_to_majority_mean'] = class_data['time_to_majority'].mean()
        network_stats.loc[network_stats['network_class'] == network_class, 'time_to_majority_median'] = class_data['time_to_majority'].median()
    
    # Flatten multiindex columns
    network_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in network_stats.columns.values]
    
    # Add metric_type column
    network_stats['metric_type'] = 'network_statistics'
    
    # 4. Similar vs opposite comparison by base algorithm
    similar_vs_opposite_base = []
    for base in ['local', 'bridge']:
        similar = non_baseline[non_baseline['friendly_name'] == f"{base} (similar)"]
        opposite = non_baseline[non_baseline['friendly_name'] == f"{base} (opposite)"]
        
        if not similar.empty and not opposite.empty:
            # Add more comprehensive metrics
            metrics = {
                'metric_type': 'similar_vs_opposite_base',
                'base_algorithm': base,
                'similar_rate_mean': similar['convergence_rate'].mean(),
                'similar_rate_median': similar['convergence_rate'].median(),
                'opposite_rate_mean': opposite['convergence_rate'].mean(),
                'opposite_rate_median': opposite['convergence_rate'].median(),
                'similar_coop_mean': similar['final_cooperativity'].mean(),
                'similar_coop_median': similar['final_cooperativity'].median(),
                'opposite_coop_mean': opposite['final_cooperativity'].mean(),
                'opposite_coop_median': opposite['final_cooperativity'].median(),
                'similar_polar_mean': similar['final_polarization'].mean(),
                'similar_polar_median': similar['final_polarization'].median(),
                'opposite_polar_mean': opposite['final_polarization'].mean(),
                'opposite_polar_median': opposite['final_polarization'].median(),
                'similar_time_mean': similar['time_to_majority'].mean(),
                'similar_time_median': similar['time_to_majority'].median(),
                'opposite_time_mean': opposite['time_to_majority'].mean(),
                'opposite_time_median': opposite['time_to_majority'].median()
            }
            
            # Calculate ratios for means
            for metric in ['rate', 'coop', 'polar', 'time']:
                sim_val = metrics[f'similar_{metric}_mean']
                opp_val = metrics[f'opposite_{metric}_mean']
                
                if not np.isnan(sim_val) and not np.isnan(opp_val) and opp_val != 0:
                    ratio = sim_val / opp_val
                    metrics[f'{metric}_ratio'] = ratio
                    metrics[f'{metric}_percent_difference'] = (ratio - 1) * 100
                    
                    # For rate/coop, higher is better; for polar/time, lower is better
                    better = 'similar' if (ratio > 1 and metric in ['rate', 'coop']) or (ratio < 1 and metric in ['polar', 'time']) else 'opposite'
                    metrics[f'{metric}_better'] = better
            
            similar_vs_opposite_base.append(metrics)
    
    # 5. Base algorithm comparison (aggregating local vs bridge) - NEW SECTION
    base_algorithm_stats = []
    for base in ['local', 'bridge']:
        base_data = non_baseline[non_baseline['algorithm_base'] == base]
        
        if not base_data.empty:
            metrics = {
                'metric_type': 'base_algorithm_statistics',
                'base_algorithm': base,
                'run_count': len(base_data),
                'convergence_rate_mean': base_data['convergence_rate'].mean(),
                'convergence_rate_median': base_data['convergence_rate'].median(),
                'final_cooperativity_mean': base_data['final_cooperativity'].mean(),
                'final_cooperativity_median': base_data['final_cooperativity'].median(),
                'final_polarization_mean': base_data['final_polarization'].mean(),
                'final_polarization_median': base_data['final_polarization'].median(),
                'time_to_majority_mean': base_data['time_to_majority'].mean(),
                'time_to_majority_median': base_data['time_to_majority'].median(),
                'better_than_static_rate_mean': base_data['better_than_static_rate'].mean(),
                'better_than_static_coop_mean': base_data['better_than_static_coop'].mean(),
                'better_than_random_rate_mean': base_data['better_than_random_rate'].mean(),
                'better_than_random_coop_mean': base_data['better_than_random_coop'].mean()
            }
            base_algorithm_stats.append(metrics)
    
    # 6. Similar vs opposite comparison (aggregated across all algorithms) - ENHANCED with median values
    similar = non_baseline[non_baseline['rewiring_type'] == 'similar']
    opposite = non_baseline[non_baseline['rewiring_type'] == 'opposite']
    
    if not similar.empty and not opposite.empty:
        # Create a comprehensive comparison of similar vs opposite strategies
        similar_vs_opposite = {
            'metric_type': 'similar_vs_opposite_overall',
            'similar_count': len(similar),
            'opposite_count': len(opposite),
            'similar_rate_mean': similar['convergence_rate'].mean(),
            'similar_rate_median': similar['convergence_rate'].median(),
            'opposite_rate_mean': opposite['convergence_rate'].mean(),
            'opposite_rate_median': opposite['convergence_rate'].median(),
            'similar_coop_mean': similar['final_cooperativity'].mean(),
            'similar_coop_median': similar['final_cooperativity'].median(),
            'opposite_coop_mean': opposite['final_cooperativity'].mean(),
            'opposite_coop_median': opposite['final_cooperativity'].median(),
            'similar_polar_mean': similar['final_polarization'].mean(),
            'similar_polar_median': similar['final_polarization'].median(),
            'opposite_polar_mean': opposite['final_polarization'].mean(),
            'opposite_polar_median': opposite['final_polarization'].median(),
            'similar_time_mean': similar['time_to_majority'].mean(),
            'similar_time_median': similar['time_to_majority'].median(),
            'opposite_time_mean': opposite['time_to_majority'].mean(),
            'opposite_time_median': opposite['time_to_majority'].median()
        }
        
        # Calculate ratios for means
        for metric in ['rate', 'coop', 'polar', 'time']:
            sim_val = similar_vs_opposite[f'similar_{metric}_mean']
            opp_val = similar_vs_opposite[f'opposite_{metric}_mean']
            
            if not np.isnan(sim_val) and not np.isnan(opp_val) and opp_val != 0:
                ratio = sim_val / opp_val
                similar_vs_opposite[f'{metric}_ratio'] = ratio
                
                # For rate/coop, higher is better; for polar/time, lower is better
                better_type = 'similar' if (ratio > 1 and metric in ['rate', 'coop']) or (ratio < 1 and metric in ['polar', 'time']) else 'opposite'
                similar_vs_opposite[f'{metric}_better'] = better_type
                
                # Calculate percentage difference
                similar_vs_opposite[f'{metric}_pct_diff'] = (ratio - 1) * 100
    else:
        similar_vs_opposite = {'metric_type': 'similar_vs_opposite_overall'}
    
    # 8. Base Algorithm Comparison (topological constraint comparison) - NEW SECTION
    base_comparison = {}
    if len(base_algorithm_stats) >= 2:
        # Get local and bridge stats
        local_stats = next((stats for stats in base_algorithm_stats if stats['base_algorithm'] == 'local'), None)
        bridge_stats = next((stats for stats in base_algorithm_stats if stats['base_algorithm'] == 'bridge'), None)
        
        if local_stats and bridge_stats:
            # Calculate deltas and percentage differences for key metrics
            base_comparison = {
                'metric_type': 'topological_constraint_comparison',
                # Convergence rate comparison
                'rate_delta': local_stats['convergence_rate_mean'] - bridge_stats['convergence_rate_mean'],
                'rate_percent_diff': ((local_stats['convergence_rate_mean'] / bridge_stats['convergence_rate_mean']) - 1) * 100 if bridge_stats['convergence_rate_mean'] != 0 else float('nan'),
                'rate_better': 'local' if local_stats['convergence_rate_mean'] > bridge_stats['convergence_rate_mean'] else 'bridge',
                
                # Cooperativity comparison
                'coop_delta': local_stats['final_cooperativity_mean'] - bridge_stats['final_cooperativity_mean'],
                'coop_percent_diff': ((local_stats['final_cooperativity_mean'] / bridge_stats['final_cooperativity_mean']) - 1) * 100 if bridge_stats['final_cooperativity_mean'] != 0 else float('nan'),
                'coop_better': 'local' if local_stats['final_cooperativity_mean'] > bridge_stats['final_cooperativity_mean'] else 'bridge',
                
                # Polarization comparison
                'polar_delta': local_stats['final_polarization_mean'] - bridge_stats['final_polarization_mean'],
                'polar_percent_diff': ((local_stats['final_polarization_mean'] / bridge_stats['final_polarization_mean']) - 1) * 100 if bridge_stats['final_polarization_mean'] != 0 else float('nan'),
                'polar_better': 'bridge' if local_stats['final_polarization_mean'] > bridge_stats['final_polarization_mean'] else 'local',
                
                # Time to majority comparison
                'time_delta': local_stats['time_to_majority_mean'] - bridge_stats['time_to_majority_mean'],
                'time_percent_diff': ((local_stats['time_to_majority_mean'] / bridge_stats['time_to_majority_mean']) - 1) * 100 if bridge_stats['time_to_majority_mean'] != 0 else float('nan'),
                'time_better': 'bridge' if local_stats['time_to_majority_mean'] > bridge_stats['time_to_majority_mean'] else 'local',
                
                # Raw values for reference
                'local_rate': local_stats['convergence_rate_mean'],
                'bridge_rate': bridge_stats['convergence_rate_mean'],
                'local_coop': local_stats['final_cooperativity_mean'],
                'bridge_coop': bridge_stats['final_cooperativity_mean'],
                'local_polar': local_stats['final_polarization_mean'],
                'bridge_polar': bridge_stats['final_polarization_mean'],
                'local_time': local_stats['time_to_majority_mean'],
                'bridge_time': bridge_stats['time_to_majority_mean']
            }
    
    # Return all summaries
    return {
        'run_stats': pd.DataFrame([run_stats]),
        'algorithm_stats': algo_stats,
        'algorithm_counts': pd.DataFrame([algo_counts]),
        'network_stats': network_stats,
        'similar_vs_opposite_base': pd.DataFrame(similar_vs_opposite_base) if similar_vs_opposite_base else pd.DataFrame([{'metric_type': 'similar_vs_opposite_base'}]),
        'similar_vs_opposite': pd.DataFrame([similar_vs_opposite]),
        'base_algorithm_stats': pd.DataFrame(base_algorithm_stats) if base_algorithm_stats else pd.DataFrame([{'metric_type': 'base_algorithm_statistics'}]),
        'topological_comparison': pd.DataFrame([base_comparison]) if base_comparison else pd.DataFrame([{'metric_type': 'topological_constraint_comparison'}])
    }


def export_combined_results(summaries, output_dir="../../Output/Stats"):
    """Export all results to a properly stacked CSV file."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, f"trajectory_metrics_{today}.csv")
    
    # Write each section sequentially with clear headers
    with open(output_path, 'w') as f:
        # 1. Individual Run Statistics
        f.write("### SECTION: Individual Run Statistics\n")
        summaries['run_stats'].to_csv(f, index=False)
        f.write('\n\n\n')
        
        # 2. Algorithm Type Statistics
        f.write("### SECTION: Algorithm Type Statistics\n")
        summaries['algorithm_counts'].to_csv(f, index=False)
        f.write('\n\n\n')
        
        # 3. Network Statistics
        f.write("### SECTION: Network Statistics\n")
        summaries['network_stats'].to_csv(f, index=False)
        f.write('\n\n\n')
        
        # 4. Algorithm Summary (details for each algorithm type)
        f.write("### SECTION: Algorithm Summary Metrics\n")
        summaries['algorithm_stats'].to_csv(f, index=False)
        f.write('\n\n\n')
        
        # 5. Similar vs Opposite Base Comparison
        f.write("### SECTION: Similar vs Opposite By Base Algorithm\n")
        summaries['similar_vs_opposite_base'].to_csv(f, index=False)
        f.write('\n\n\n')
        
        # 6. Similar vs Opposite Overall Comparison
        f.write("### SECTION: Similar vs Opposite Overall Comparison\n")
        summaries['similar_vs_opposite'].to_csv(f, index=False)
        f.write('\n\n\n')
        
        # 7. Base Algorithm Statistics (NEW)
        f.write("### SECTION: Base Algorithm Statistics\n")
        summaries['base_algorithm_stats'].to_csv(f, index=False)
        f.write('\n\n\n')
        
        # 8. Topological Constraint Comparison (NEW)
        f.write("### SECTION: Topological Constraint Comparison\n")
        summaries['topological_comparison'].to_csv(f, index=False)
    
    return output_path


def main():
    """Main execution function."""
    # Get data file path
    input_dir = "../../Output"
    file_list = [f for f in os.listdir(input_dir) if "default_run_individual" in f and f.endswith(".csv")]
    
    if not file_list:
        print("No individual run files found in the output directory.")
        return
    
    # List available files
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    # Get user selection
    file_index = int(input(f"Enter the index of the file to analyze (0-{len(file_list)-1}): "))
    if file_index < 0 or file_index >= len(file_list):
        print("Invalid selection.")
        return
        
    file_path = os.path.join(input_dir, file_list[file_index])
    
    # Analyze data
    t_max = 40000  # Maximum timestep to analyze
    summaries = analyze_trajectory_data(file_path, t_max)
    
    if summaries is None:
        print("Analysis failed.")
        return
    
    # Export results
    output_path = export_combined_results(summaries)
    
    # Print key statistics
    algorithm_counts = summaries['algorithm_counts'].iloc[0]
    run_statistics = summaries['run_stats'].iloc[0]
    
    print("\nKey Statistics:")
    print(f"- Individual runs: {run_statistics['total_runs']}")
    print(f"  Better than static (rate): {run_statistics['vs_static_rate_count']} ({run_statistics['vs_static_rate_pct']:.1f}%)")
    print(f"  Better than random (rate): {run_statistics['vs_random_rate_count']} ({run_statistics['vs_random_rate_pct']:.1f}%)")
    
    print(f"- Algorithm types: {algorithm_counts['total_algorithm_types']}")
    print(f"  Better than static (rate): {algorithm_counts['algo_better_static_rate']} ({algorithm_counts['algo_better_static_rate_pct']:.1f}%)")
    print(f"  Better than random (rate): {algorithm_counts['algo_better_random_rate']} ({algorithm_counts['algo_better_random_rate_pct']:.1f}%)")
    
    # Print similar vs opposite comparison if available
    if 'similar_time_mean' in summaries['similar_vs_opposite'].columns:
        similar_vs_opposite = summaries['similar_vs_opposite'].iloc[0]
        time_ratio = similar_vs_opposite.get('time_ratio', np.nan)
        if not np.isnan(time_ratio):
            faster = "opposite" if time_ratio > 1 else "similar"
            pct_diff = abs(similar_vs_opposite.get('time_pct_diff', 0))
            print(f"\n- Strategy comparison: {faster} strategy is {pct_diff:.1f}% faster to reach cooperative majority")
    
    # Print base algorithm comparison if available
    if len(summaries['base_algorithm_stats']) > 1:
        print("\n- Base Algorithm Comparison:")
        for _, row in summaries['base_algorithm_stats'].iterrows():
            if row['base_algorithm'] in ['local', 'bridge']:
                print(f"  {row['base_algorithm'].capitalize()}: " +
                     f"Coop {row['final_cooperativity_mean']:.3f}, " +
                     f"Conv. Rate {row['convergence_rate_mean']:.3f}")
                     
    # Print topological constraint comparison if available
    if 'topological_comparison' in summaries and not summaries['topological_comparison'].empty:
        topo_comp = summaries['topological_comparison'].iloc[0]
        if 'rate_delta' in topo_comp:
            print("\n- Topological Constraint Comparison:")
            print(f"  Convergence Rate Delta: {topo_comp['rate_delta']:.3f} ({abs(topo_comp['rate_percent_diff']):.1f}% {'higher' if topo_comp['rate_percent_diff'] > 0 else 'lower'} for local)")
            print(f"  Final Cooperativity Delta: {topo_comp['coop_delta']:.3f} ({abs(topo_comp['coop_percent_diff']):.1f}% {'higher' if topo_comp['coop_percent_diff'] > 0 else 'lower'} for local)")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()