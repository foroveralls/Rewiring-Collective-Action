import pandas as pd
import numpy as np
import os
from datetime import date

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

def get_friendly_name(scenario):
    parts = scenario.split()
    key = f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{parts[0]}_none"
    return FRIENDLY_NAMES.get(key, scenario)

def calculate_max_backfirer_fraction(scenario_data):
    """
    Find the highest polarizing node fraction that still results in a cooperative state.
    """
    cooperative_fractions = []
    for fraction in scenario_data['polarisingNode_f'].unique():
        fraction_data = scenario_data[scenario_data['polarisingNode_f'] == fraction]
        if (fraction_data['state'] > 0).any():
            cooperative_fractions.append(fraction)
    
    return max(cooperative_fractions) if cooperative_fractions else 0.0

def calculate_metrics(df):
    """
    Calculate comprehensive metrics across different scenarios, 
    aggregating over polarisingNode_f and other parameters.
    """
    all_metrics = []
    
    # Group by topology and scenario
    for (topology, scenario), group in df.groupby(['topology', 'scenario']):
        friendly_name = get_friendly_name(scenario)
        
        # Get positive cooperation states
        positive_states = group[group['state'] > 0]
        
        # Calculate backfirer metrics only for cooperative outcomes
        backfirer_fractions = positive_states['polarisingNode_f'].values
        
        metrics = {
            'topology': topology,
            'friendly_name': friendly_name,
            
            # Overall state metrics
            'median_cooperation': group['state'].median(),
            'mean_cooperation': group['state'].mean(),
            'std_cooperation': group['state'].std(),
            'max_cooperation': group['state'].max(),
            'min_cooperation': group['state'].min(),
            'cooperative_ratio': (group['state'] > 0).mean(),
            
            # Positive cooperation metrics
            'median_positive_cooperation': positive_states['state'].median() if len(positive_states) > 0 else np.nan,
            'mean_positive_cooperation': positive_states['state'].mean() if len(positive_states) > 0 else np.nan,
            
            # Backfirer metrics - only consider cases that led to cooperation
            'max_backfirer_fraction': np.max(backfirer_fractions) if len(backfirer_fractions) > 0 else 0.0,
            'mean_backfirer_fraction': np.mean(backfirer_fractions) if len(backfirer_fractions) > 0 else 0.0,
            'median_backfirer_fraction': np.median(backfirer_fractions) if len(backfirer_fractions) > 0 else 0.0,
            'backfirer_fraction_std': np.std(backfirer_fractions) if len(backfirer_fractions) > 0 else 0.0,
        }
        
        all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)

def calculate_summary_metrics(metrics_df):
    """
    Calculate separate summaries by topology and by rewiring strategy.
    Only include cases where cooperation occurred.
    """
    # Filter for meaningful backfirer fractions
    valid_metrics = metrics_df[metrics_df['max_backfirer_fraction'] > 0].copy()
    
    # Calculate metrics for topology summary
    topology_summary = valid_metrics.groupby('topology').agg({
        'mean_cooperation': 'mean',
        'median_cooperation': 'median',
        'std_cooperation': 'mean',
        'cooperative_ratio': 'mean',
        'mean_backfirer_fraction': 'mean',
        'max_backfirer_fraction': 'max',
        'backfirer_fraction_std': 'std'
    }).round(3)
    
    # Calculate metrics for strategy summary
    strategy_summary = valid_metrics.groupby('friendly_name').agg({
        'mean_cooperation': 'mean',
        'median_cooperation': 'median',
        'std_cooperation': 'mean',
        'cooperative_ratio': 'mean',
        'mean_backfirer_fraction': 'mean',
        'max_backfirer_fraction': 'max',
        'backfirer_fraction_std': 'std'
    }).round(3)
    
    return topology_summary, strategy_summary


def save_metrics(metrics_df, topology_summary, strategy_summary, output_dir='../Output'):
    """Save all metrics to a single file with clear sections."""
    today = date.today().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, f'heatmap_metrics_{today}.csv')
    
    # Round numerical columns
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    metrics_df[numeric_cols] = metrics_df[numeric_cols].round(3)
    
    with open(output_path, 'w') as f:
        # Write main metrics
        f.write("Main Metrics:\n")
        metrics_df.to_csv(f, index=False)
        f.write('\n\n')
        
        # Write topology summary
        f.write("Summary by Topology:\n")
        topology_summary.to_csv(f)
        f.write('\n\n')
        
        # Write strategy summary
        f.write("Summary by Rewiring Strategy:\n")
        strategy_summary.to_csv(f)
    
    return output_path

def main():
    """Main execution function."""
    # Get data file path
    file_list = [f for f in os.listdir("../Output") if f.endswith(".csv") and "heatmap" in f]
    
    if not file_list:
        print("No suitable files found in the Output directory.")
        return
    
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    file_index = int(input("Enter the index of the file you want to analyze: "))
    data_path = os.path.join("../Output", file_list[file_index])
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    df.loc[df['mode'].isin(['wtf', 'node2vec']), 'rewiring'] = 'empirical'
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    df['scenario'] = df['rewiring'] + ' ' + df['mode']
    
    print("\nCalculating metrics...")
    metrics_df = calculate_metrics(df)
    
    # Calculate backfirer metrics for cooperative states
    cooperative_data = metrics_df[metrics_df['cooperative_ratio'] > 0].copy()
    topology_summary, strategy_summary = calculate_summary_metrics(cooperative_data)
    
    # Save results
    output_path = save_metrics(metrics_df, topology_summary, strategy_summary)
    print(f"\nMetrics saved to: {output_path}")
    
    return metrics_df, topology_summary, strategy_summary

if __name__ == "__main__":
    main()