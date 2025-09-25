#!/usr/bin/env python3
"""
Fixed backfirer regime analysis that includes ALL data points, not just cooperative ones
This provides the full backfirer parameter range (0.0-1.0) for regime analysis
"""
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

TARGET_TOPOLOGIES = ['FB', 'Twitter', 'cl', 'DPAH']

STUBBORNNESS_REGIMES = {
    'low': (0.0, 0.4),
    'medium': (0.4, 0.7), 
    'high': (0.7, 1.0)
}

def get_friendly_name(scenario):
    parts = scenario.split()
    key = f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{parts[0]}_none"
    return FRIENDLY_NAMES.get(key, scenario)

def calculate_variant_type(friendly_name):
    if 'opposite' in friendly_name:
        return 'opposite'
    elif 'similar' in friendly_name:
        return 'similar'
    else:
        return 'other'

def get_stubbornness_regime(stubbornness_value):
    for regime, (min_val, max_val) in STUBBORNNESS_REGIMES.items():
        if min_val <= stubbornness_value < max_val:
            return regime
    return 'high'  # Handle edge case where stubbornness = 1.0

def calculate_fixed_regime_metrics(df):
    """Calculate regime metrics with full backfirer data (not limited to cooperative states)"""
    df = df[df['topology'].isin(TARGET_TOPOLOGIES)].copy()
    df['stubbornness_regime'] = df['stubbornness'].apply(get_stubbornness_regime)
    
    all_metrics = []
    
    for (topology, scenario, regime), group in df.groupby(['topology', 'scenario', 'stubbornness_regime']):
        friendly_name = get_friendly_name(scenario)
        variant_type = calculate_variant_type(friendly_name)
        
        total_combinations = len(group)
        cooperative_mask = group['state'] > 0
        cooperative_data = group[cooperative_mask]
        n_cooperative = len(cooperative_data)
        
        # KEY FIX: Calculate backfirer metrics on ALL data points, not just cooperative ones
        all_backfirer_fractions = group['polarisingNode_f'].values
        coop_backfirer_fractions = cooperative_data['polarisingNode_f'].values
        
        metrics = {
            'topology': topology,
            'scenario': scenario,
            'friendly_name': friendly_name,
            'variant_type': variant_type,
            'stubbornness_regime': regime,
            
            # Performance metrics
            'mean_cooperation': group['state'].mean(),
            'median_cooperation': group['state'].median(),
            'cooperative_ratio': n_cooperative / total_combinations if total_combinations > 0 else 0,
            'cooperative_volume_percent': (n_cooperative / total_combinations) * 100 if total_combinations > 0 else 0,
            'mean_cooperation_coop_only': cooperative_data['state'].mean() if n_cooperative > 0 else np.nan,
            
            # FIXED: Backfirer metrics on ALL data points (full parameter space)
            'max_backfirer_fraction_all': np.max(all_backfirer_fractions) if len(all_backfirer_fractions) > 0 else 0.0,
            'min_backfirer_fraction_all': np.min(all_backfirer_fractions) if len(all_backfirer_fractions) > 0 else 0.0,
            'mean_backfirer_fraction_all': np.mean(all_backfirer_fractions) if len(all_backfirer_fractions) > 0 else 0.0,
            'median_backfirer_fraction_all': np.median(all_backfirer_fractions) if len(all_backfirer_fractions) > 0 else 0.0,
            
            # Traditional backfirer metrics (cooperative states only - for comparison)
            'max_backfirer_fraction_coop': np.max(coop_backfirer_fractions) if len(coop_backfirer_fractions) > 0 else 0.0,
            'mean_backfirer_fraction_coop': np.mean(coop_backfirer_fractions) if len(coop_backfirer_fractions) > 0 else 0.0,
            
            # Polarization metrics
            'mean_polarization': group['state_std'].mean(),
            'high_polarization_percent': (len(group[group['state_std'] >= 0.8]) / total_combinations) * 100 if total_combinations > 0 else 0,
            
            # Sample sizes
            'n_combinations': total_combinations,
            'n_cooperative': n_cooperative
        }
        
        all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)

def create_comprehensive_algorithm_comparison_fixed(regime_metrics_df):
    """Create comprehensive comparison tables using the fixed backfirer metrics"""
    
    comparison_tables = {}
    
    for regime in ['low', 'medium', 'high']:
        regime_data = regime_metrics_df[regime_metrics_df['stubbornness_regime'] == regime]
        
        if len(regime_data) == 0:
            continue
            
        # Calculate algorithm-level aggregations using the fixed metrics
        algorithm_stats = {}
        
        algorithm_mapping = {
            'local (opposite)': 'Opposite',
            'local (similar)': 'Similar', 
            'wtf': 'WTF',
            'node2vec': 'Node2Vec',
            'static': 'Static',
            'random': 'Random'
        }
        
        # For WTF and Node2Vec, only include topologies where they exist
        for original_name, display_name in algorithm_mapping.items():
            algo_data = regime_data[regime_data['friendly_name'] == original_name]
            
            if len(algo_data) > 0:
                # Weight calculations by number of combinations per topology to get proper averages
                total_combinations = algo_data['n_combinations'].sum()
                total_cooperative = algo_data['n_cooperative'].sum()
                
                # Weighted averages for metrics that should be topology-weighted
                cooperation_weighted = (algo_data['mean_cooperation'] * algo_data['n_combinations']).sum() / total_combinations
                polarization_weighted = (algo_data['mean_polarization'] * algo_data['n_combinations']).sum() / total_combinations
                
                # For coop-only metrics, weight by cooperative combinations
                if total_cooperative > 0:
                    coop_only_weighted = (algo_data['mean_cooperation_coop_only'] * algo_data['n_cooperative']).sum() / total_cooperative
                else:
                    coop_only_weighted = np.nan
                
                algorithm_stats[display_name] = {
                    'mean_cooperation': cooperation_weighted,
                    'mean_cooperation_coop_only': coop_only_weighted,
                    'cooperative_volume_percent': (total_cooperative / total_combinations) * 100 if total_combinations > 0 else 0,
                    # Use the FIXED backfirer metrics (all data points)
                    'max_backfirer_fraction': algo_data['max_backfirer_fraction_all'].max(),
                    'mean_backfirer_fraction': (algo_data['mean_backfirer_fraction_all'] * algo_data['n_combinations']).sum() / total_combinations,
                    'mean_polarization': polarization_weighted,
                    'high_polarization_percent': (algo_data['high_polarization_percent'] * algo_data['n_combinations']).sum() / total_combinations,
                    'n_topologies': len(algo_data['topology'].unique()),
                    'total_combinations': total_combinations
                }
        
        # Create comparison table with consistent column order
        comparison_rows = []
        metrics_to_compare = [
            ('Mean Cooperation', 'mean_cooperation'),
            ('Mean Coop (Coop Only)', 'mean_cooperation_coop_only'),
            ('Cooperative Volume %', 'cooperative_volume_percent'),
            ('Max Backfirer Fraction', 'max_backfirer_fraction'),
            ('Mean Backfirer Fraction', 'mean_backfirer_fraction'),
            ('High Polarization %', 'high_polarization_percent'),
            ('Mean Polarization', 'mean_polarization')
        ]
        
        # Consistent column order
        column_order = ['Metric', 'Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
        
        for metric_name, metric_key in metrics_to_compare:
            row = {'Metric': metric_name}
            for col in column_order[1:]:  # Skip 'Metric'
                if col in algorithm_stats and metric_key in algorithm_stats[col]:
                    row[col] = algorithm_stats[col][metric_key]
                else:
                    row[col] = np.nan
            comparison_rows.append(row)
        
        comparison_tables[regime] = pd.DataFrame(comparison_rows)[column_order]
    
    return comparison_tables

def main():
    """Generate fixed backfirer regime analysis"""
    file_path = "../../Output/heatmap_sweep_phased_sweep_20250904_1545_stubbornness_polarisingNode_f_ihp.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print("Loading and processing data with FIXED backfirer regime analysis...")
    
    # Load and prepare data  
    df = pd.read_csv(file_path)
    df.loc[df['mode'].isin(['wtf', 'node2vec']), 'rewiring'] = 'empirical'
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    df['scenario'] = df['rewiring'] + ' ' + df['mode']
    
    print("Calculating fixed regime metrics (using ALL data points for backfirer analysis)...")
    
    # Calculate fixed metrics
    regime_metrics_df = calculate_fixed_regime_metrics(df)
    
    # Create comprehensive comparison tables with fixed backfirer metrics
    comparison_tables = create_comprehensive_algorithm_comparison_fixed(regime_metrics_df)
    
    # Save results
    output_dir = "../../Output/Stats/stubborness_backfirer"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    
    saved_files = []
    
    for regime, table in comparison_tables.items():
        file_path = os.path.join(output_dir, f'comprehensive_algorithm_comparison_{regime}_FIXED_{today}.csv')
        table.round(3).to_csv(file_path, index=False)
        saved_files.append(file_path)
        print(f"Saved fixed {regime} regime comparison: {file_path}")
    
    # Print comparison of old vs new backfirer ranges
    print(f"\n=== BACKFIRER RANGE COMPARISON (Old vs Fixed) ===")
    
    for regime in ['low', 'medium', 'high']:
        regime_data = regime_metrics_df[regime_metrics_df['stubbornness_regime'] == regime]
        if len(regime_data) > 0:
            old_max = regime_data['max_backfirer_fraction_coop'].max()
            new_max = regime_data['max_backfirer_fraction_all'].max()
            old_min = regime_data['min_backfirer_fraction_all'].min()  # This should be same for both
            
            print(f"\n{regime.upper()} regime:")
            print(f"  Old range (coop-only): {regime_data['max_backfirer_fraction_coop'].min():.3f} - {old_max:.3f}")
            print(f"  New range (all data):  {old_min:.3f} - {new_max:.3f}")
            print(f"  Improvement: +{new_max - old_max:.3f} additional backfirer coverage")
    
    print(f"\nFixed backfirer regime analysis complete!")
    return regime_metrics_df, comparison_tables, saved_files

if __name__ == "__main__":
    main()