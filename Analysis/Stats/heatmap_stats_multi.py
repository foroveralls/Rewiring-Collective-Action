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
    'node2vec_none': 'node2vec',
    'wtf_empirical': 'wtf',  # Handle 'empirical wtf' scenario
    'node2vec_empirical': 'node2vec'  # Handle 'empirical node2vec' scenario
}

# Include all topologies
TARGET_TOPOLOGIES = ['FB', 'Twitter', 'cl', 'DPAH']

def get_friendly_name(scenario):
    parts = scenario.split()
    key = f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{parts[0]}_none"
    return FRIENDLY_NAMES.get(key, scenario)

def calculate_variant_type(friendly_name):
    """Classify as opposite, similar, or other"""
    if 'opposite' in friendly_name:
        return 'opposite'
    elif 'similar' in friendly_name:
        return 'similar'
    else:
        return 'other'

def calculate_metrics(df):
    """Calculate comprehensive metrics for FB/Twitter networks"""
    # Filter to target topologies
    df = df[df['topology'].isin(TARGET_TOPOLOGIES)].copy()
    
    all_metrics = []
    
    for (topology, scenario), group in df.groupby(['topology', 'scenario']):
        friendly_name = get_friendly_name(scenario)
        variant_type = calculate_variant_type(friendly_name)
        
        # Total parameter combinations
        total_combinations = len(group)
        
        # Cooperative states (x > 0)
        cooperative_mask = group['state'] > 0
        cooperative_data = group[cooperative_mask]
        n_cooperative = len(cooperative_data)
        
        # High polarization states (σ(x) >= 0.8)
        high_polarization_mask = group['state_std'] >= 0.8
        n_high_polarization = len(group[high_polarization_mask])
        
        # Backfirer analysis for cooperative states only
        coop_backfirer_fractions = cooperative_data['polarisingNode_f'].values
        coop_stubbornness = cooperative_data['stubbornness'].values
        
        # Stubbornness sensitivity analysis - Sobol-style main effect calculation
        stub_mean_coop_by_level = []
        if 'stubbornness' in group.columns:
            for stub_level in group['stubbornness'].unique():
                stub_data = group[group['stubbornness'] == stub_level]
                if len(stub_data) > 0:
                    # Average cooperation across all backfirer levels for this stubbornness level
                    stub_mean_coop_by_level.append(stub_data['state'].mean())
        
        # Calculate sensitivity as standard deviation of mean cooperation per parameter level
        stub_sensitivity = np.std(stub_mean_coop_by_level) if len(stub_mean_coop_by_level) > 1 else 0.0
        
        # Backfirer sensitivity analysis - Sobol-style main effect calculation  
        backf_mean_coop_by_level = []
        if 'polarisingNode_f' in group.columns:
            for backf_level in group['polarisingNode_f'].unique():
                backf_data = group[group['polarisingNode_f'] == backf_level]
                if len(backf_data) > 0:
                    # Average cooperation across all stubbornness levels for this backfirer level
                    backf_mean_coop_by_level.append(backf_data['state'].mean())
        
        # Calculate sensitivity as standard deviation of mean cooperation per parameter level
        backf_sensitivity = np.std(backf_mean_coop_by_level) if len(backf_mean_coop_by_level) > 1 else 0.0
        
        # Supplementary variance decomposition analysis
        # Calculate within-level variance (noise) and between-level variance (signal)
        
        # Stubbornness variance decomposition
        stub_within_variance = 0.0
        stub_signal_to_noise = 0.0
        if len(stub_mean_coop_by_level) > 1:
            stub_level_variances = []
            for stub_level in group['stubbornness'].unique():
                stub_data = group[group['stubbornness'] == stub_level]['state'].values
                if len(stub_data) > 1:
                    stub_level_variances.append(np.var(stub_data, ddof=1))
            
            if stub_level_variances:
                stub_within_variance = np.mean(stub_level_variances)  # Average within-level variance
                stub_between_variance = np.var(stub_mean_coop_by_level, ddof=1)  # Between-level variance
                stub_signal_to_noise = stub_between_variance / stub_within_variance if stub_within_variance > 0 else 0.0
        
        # Backfirer variance decomposition  
        backf_within_variance = 0.0
        backf_signal_to_noise = 0.0
        if len(backf_mean_coop_by_level) > 1:
            backf_level_variances = []
            for backf_level in group['polarisingNode_f'].unique():
                backf_data = group[group['polarisingNode_f'] == backf_level]['state'].values
                if len(backf_data) > 1:
                    backf_level_variances.append(np.var(backf_data, ddof=1))
            
            if backf_level_variances:
                backf_within_variance = np.mean(backf_level_variances)  # Average within-level variance
                backf_between_variance = np.var(backf_mean_coop_by_level, ddof=1)  # Between-level variance
                backf_signal_to_noise = backf_between_variance / backf_within_variance if backf_within_variance > 0 else 0.0
        
        metrics = {
            'topology': topology,
            'scenario': scenario,
            'friendly_name': friendly_name,
            'variant_type': variant_type,
            
            # Overall cooperation metrics
            'mean_cooperation': group['state'].mean(),
            'median_cooperation': group['state'].median(),
            'cooperative_ratio': n_cooperative / total_combinations,
            'cooperative_volume_percent': (n_cooperative / total_combinations) * 100,
            
            # Cooperative-only metrics
            'mean_cooperation_coop_only': cooperative_data['state'].mean() if n_cooperative > 0 else np.nan,
            'median_cooperation_coop_only': cooperative_data['state'].median() if n_cooperative > 0 else np.nan,
            
            # Backfirer metrics (cooperative states only)
            'max_backfirer_fraction': np.max(coop_backfirer_fractions) if len(coop_backfirer_fractions) > 0 else 0.0,
            'min_backfirer_fraction': np.min(coop_backfirer_fractions) if len(coop_backfirer_fractions) > 0 else 0.0,
            'mean_backfirer_fraction': np.mean(coop_backfirer_fractions) if len(coop_backfirer_fractions) > 0 else 0.0,
            'median_backfirer_fraction': np.median(coop_backfirer_fractions) if len(coop_backfirer_fractions) > 0 else 0.0,
            
            # Stubbornness metrics (cooperative states only)
            'max_stubbornness_coop': np.max(coop_stubbornness) if len(coop_stubbornness) > 0 else 0.0,
            'min_stubbornness_coop': np.min(coop_stubbornness) if len(coop_stubbornness) > 0 else 0.0,
            'mean_stubbornness_coop': np.mean(coop_stubbornness) if len(coop_stubbornness) > 0 else 0.0,
            
            # Sensitivity metrics (Sobol main effects - higher = more sensitive to parameter changes)
            'stubbornness_sensitivity': stub_sensitivity,  # std of mean cooperation across stubbornness levels
            'backfirer_sensitivity': backf_sensitivity,    # std of mean cooperation across backfirer levels
            
            # Supplementary variance decomposition metrics
            'stubbornness_within_variance': stub_within_variance,     # Average variance within parameter levels (noise)
            'stubbornness_signal_to_noise': stub_signal_to_noise,     # Between-level variance / within-level variance
            'backfirer_within_variance': backf_within_variance,       # Average variance within parameter levels (noise) 
            'backfirer_signal_to_noise': backf_signal_to_noise,       # Between-level variance / within-level variance
            
            # Polarization metrics
            'high_polarization_percent': (n_high_polarization / total_combinations) * 100,
            'mean_polarization': group['state_std'].mean(),
            
            # Ranges
            'backfirer_range': np.max(coop_backfirer_fractions) - np.min(coop_backfirer_fractions) if len(coop_backfirer_fractions) > 1 else 0.0,
            'stubbornness_range': np.max(coop_stubbornness) - np.min(coop_stubbornness) if len(coop_stubbornness) > 1 else 0.0,
            'total_combinations': total_combinations,
            'n_cooperative': n_cooperative
        }
        
        all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)

def calculate_variant_comparison(metrics_df):
    """Compare opposite vs similar vs WTF vs N2V variants"""
    comparison_data = []
    
    for topology in TARGET_TOPOLOGIES:
        topo_data = metrics_df[metrics_df['topology'] == topology]
        
        opposite_data = topo_data[topo_data['variant_type'] == 'opposite']
        similar_data = topo_data[topo_data['variant_type'] == 'similar']
        wtf_data = topo_data[topo_data['friendly_name'] == 'wtf']
        n2v_data = topo_data[topo_data['friendly_name'] == 'node2vec']
        static_data = topo_data[topo_data['friendly_name'] == 'static']
        random_data = topo_data[topo_data['friendly_name'] == 'random']
        
        if len(opposite_data) > 0 and len(similar_data) > 0:
            comparison = {
                'topology': topology,
                'opposite_mean_coop': opposite_data['mean_cooperation'].mean(),
                'similar_mean_coop': similar_data['mean_cooperation'].mean(),
                'opposite_coop_only_mean': opposite_data['mean_cooperation_coop_only'].mean(),
                'similar_coop_only_mean': similar_data['mean_cooperation_coop_only'].mean(),
                
                # Backfirer tolerance and ranges
                'opposite_max_backfirer': opposite_data['max_backfirer_fraction'].max(),
                'similar_max_backfirer': similar_data['max_backfirer_fraction'].max(),
                'opposite_backfirer_range': f"[{opposite_data['min_backfirer_fraction'].min():.2f}-{opposite_data['max_backfirer_fraction'].max():.2f}]",
                'similar_backfirer_range': f"[{similar_data['min_backfirer_fraction'].min():.2f}-{similar_data['max_backfirer_fraction'].max():.2f}]",
                
                # Sensitivity metrics (lower = less sensitive)
                'opposite_stub_sensitivity': opposite_data['stubbornness_sensitivity'].mean(),
                'similar_stub_sensitivity': similar_data['stubbornness_sensitivity'].mean(),
                'opposite_backf_sensitivity': opposite_data['backfirer_sensitivity'].mean(), 
                'similar_backf_sensitivity': similar_data['backfirer_sensitivity'].mean(),
                
                # Volume metrics (calculate as percentage of total parameter combinations)
                'opposite_coop_volume': (opposite_data['n_cooperative'].sum() / opposite_data['total_combinations'].sum()) * 100,
                'similar_coop_volume': (similar_data['n_cooperative'].sum() / similar_data['total_combinations'].sum()) * 100,
                
                # Polarization metrics
                'opposite_high_pol_percent': opposite_data['high_polarization_percent'].mean(),
                'similar_high_pol_percent': similar_data['high_polarization_percent'].mean(),
                'opposite_mean_polarization': opposite_data['mean_polarization'].mean(),
                'similar_mean_polarization': similar_data['mean_polarization'].mean(),
                
                # Stubbornness ranges for cooperative states
                'opposite_stub_range': f"[{opposite_data['min_stubbornness_coop'].min():.2f}-{opposite_data['max_stubbornness_coop'].max():.2f}]",
                'similar_stub_range': f"[{similar_data['min_stubbornness_coop'].min():.2f}-{similar_data['max_stubbornness_coop'].max():.2f}]",
                
                # WTF metrics
                'wtf_mean_coop': wtf_data['mean_cooperation'].mean() if len(wtf_data) > 0 else np.nan,
                'wtf_coop_only_mean': wtf_data['mean_cooperation_coop_only'].mean() if len(wtf_data) > 0 else np.nan,
                'wtf_max_backfirer': wtf_data['max_backfirer_fraction'].max() if len(wtf_data) > 0 else np.nan,
                'wtf_backfirer_range': f"[{wtf_data['min_backfirer_fraction'].min():.2f}-{wtf_data['max_backfirer_fraction'].max():.2f}]" if len(wtf_data) > 0 else "N/A",
                'wtf_stub_sensitivity': wtf_data['stubbornness_sensitivity'].mean() if len(wtf_data) > 0 else np.nan,
                'wtf_backf_sensitivity': wtf_data['backfirer_sensitivity'].mean() if len(wtf_data) > 0 else np.nan,
                'wtf_coop_volume': (wtf_data['n_cooperative'].sum() / wtf_data['total_combinations'].sum()) * 100 if len(wtf_data) > 0 else np.nan,
                'wtf_high_pol_percent': wtf_data['high_polarization_percent'].mean() if len(wtf_data) > 0 else np.nan,
                'wtf_mean_polarization': wtf_data['mean_polarization'].mean() if len(wtf_data) > 0 else np.nan,
                'wtf_stub_range': f"[{wtf_data['min_stubbornness_coop'].min():.2f}-{wtf_data['max_stubbornness_coop'].max():.2f}]" if len(wtf_data) > 0 else "N/A",
                
                # Node2Vec metrics
                'n2v_mean_coop': n2v_data['mean_cooperation'].mean() if len(n2v_data) > 0 else np.nan,
                'n2v_coop_only_mean': n2v_data['mean_cooperation_coop_only'].mean() if len(n2v_data) > 0 else np.nan,
                'n2v_max_backfirer': n2v_data['max_backfirer_fraction'].max() if len(n2v_data) > 0 else np.nan,
                'n2v_backfirer_range': f"[{n2v_data['min_backfirer_fraction'].min():.2f}-{n2v_data['max_backfirer_fraction'].max():.2f}]" if len(n2v_data) > 0 else "N/A",
                'n2v_stub_sensitivity': n2v_data['stubbornness_sensitivity'].mean() if len(n2v_data) > 0 else np.nan,
                'n2v_backf_sensitivity': n2v_data['backfirer_sensitivity'].mean() if len(n2v_data) > 0 else np.nan,
                'n2v_coop_volume': (n2v_data['n_cooperative'].sum() / n2v_data['total_combinations'].sum()) * 100 if len(n2v_data) > 0 else np.nan,
                'n2v_high_pol_percent': n2v_data['high_polarization_percent'].mean() if len(n2v_data) > 0 else np.nan,
                'n2v_mean_polarization': n2v_data['mean_polarization'].mean() if len(n2v_data) > 0 else np.nan,
                'n2v_stub_range': f"[{n2v_data['min_stubbornness_coop'].min():.2f}-{n2v_data['max_stubbornness_coop'].max():.2f}]" if len(n2v_data) > 0 else "N/A",
                
                # Static metrics
                'static_mean_coop': static_data['mean_cooperation'].mean() if len(static_data) > 0 else np.nan,
                'static_coop_only_mean': static_data['mean_cooperation_coop_only'].mean() if len(static_data) > 0 else np.nan,
                'static_max_backfirer': static_data['max_backfirer_fraction'].max() if len(static_data) > 0 else np.nan,
                'static_backfirer_range': f"[{static_data['min_backfirer_fraction'].min():.2f}-{static_data['max_backfirer_fraction'].max():.2f}]" if len(static_data) > 0 else "N/A",
                'static_stub_sensitivity': static_data['stubbornness_sensitivity'].mean() if len(static_data) > 0 else np.nan,
                'static_backf_sensitivity': static_data['backfirer_sensitivity'].mean() if len(static_data) > 0 else np.nan,
                'static_coop_volume': (static_data['n_cooperative'].sum() / static_data['total_combinations'].sum()) * 100 if len(static_data) > 0 else np.nan,
                'static_high_pol_percent': static_data['high_polarization_percent'].mean() if len(static_data) > 0 else np.nan,
                'static_mean_polarization': static_data['mean_polarization'].mean() if len(static_data) > 0 else np.nan,
                'static_stub_range': f"[{static_data['min_stubbornness_coop'].min():.2f}-{static_data['max_stubbornness_coop'].max():.2f}]" if len(static_data) > 0 else "N/A",
                
                # Random metrics
                'random_mean_coop': random_data['mean_cooperation'].mean() if len(random_data) > 0 else np.nan,
                'random_coop_only_mean': random_data['mean_cooperation_coop_only'].mean() if len(random_data) > 0 else np.nan,
                'random_max_backfirer': random_data['max_backfirer_fraction'].max() if len(random_data) > 0 else np.nan,
                'random_backfirer_range': f"[{random_data['min_backfirer_fraction'].min():.2f}-{random_data['max_backfirer_fraction'].max():.2f}]" if len(random_data) > 0 else "N/A",
                'random_stub_sensitivity': random_data['stubbornness_sensitivity'].mean() if len(random_data) > 0 else np.nan,
                'random_backf_sensitivity': random_data['backfirer_sensitivity'].mean() if len(random_data) > 0 else np.nan,
                'random_coop_volume': (random_data['n_cooperative'].sum() / random_data['total_combinations'].sum()) * 100 if len(random_data) > 0 else np.nan,
                'random_high_pol_percent': random_data['high_polarization_percent'].mean() if len(random_data) > 0 else np.nan,
                'random_mean_polarization': random_data['mean_polarization'].mean() if len(random_data) > 0 else np.nan,
                'random_stub_range': f"[{random_data['min_stubbornness_coop'].min():.2f}-{random_data['max_stubbornness_coop'].max():.2f}]" if len(random_data) > 0 else "N/A"
            }
            comparison_data.append(comparison)
    
    return pd.DataFrame(comparison_data)

def calculate_directed_networks_comparison(metrics_df):
    """Head-to-head comparison on directed networks only (DPAH, Twitter) where WTF is applicable"""
    # Directed networks where WTF can be applied
    directed_topologies = ['DPAH', 'Twitter']
    directed_data = metrics_df[metrics_df['topology'].isin(directed_topologies)].copy()

    print(f"\n=== DIRECTED NETWORKS HEAD-TO-HEAD COMPARISON ===")
    print(f"Comparing all algorithms on directed networks only: {directed_topologies}")
    print("This provides fair comparison since WTF only works on directed networks")
    print(f"\nTotal records in directed networks: {len(directed_data)}")
    print(f"Topologies present: {directed_data['topology'].unique()}")
    print(f"Scenarios present: {directed_data['scenario'].unique()}")
    print(f"Friendly names present: {directed_data['friendly_name'].unique()}")

    # Group algorithms by type
    opposite_data = directed_data[directed_data['variant_type'] == 'opposite']
    similar_data = directed_data[directed_data['variant_type'] == 'similar']
    wtf_data = directed_data[directed_data['friendly_name'] == 'wtf']
    n2v_data = directed_data[directed_data['friendly_name'] == 'node2vec']
    static_data = directed_data[directed_data['friendly_name'] == 'static']
    random_data = directed_data[directed_data['friendly_name'] == 'random']

    print(f"\nData counts:")
    print(f"  Opposite: {len(opposite_data)}")
    print(f"  Similar: {len(similar_data)}")
    print(f"  WTF: {len(wtf_data)}")
    print(f"  Node2Vec: {len(n2v_data)}")
    print(f"  Static: {len(static_data)}")
    print(f"  Random: {len(random_data)}")
    
    directed_comparison = {}

    # Only add data for algorithms that have records
    if len(opposite_data) > 0:
        directed_comparison['opposite'] = {
            'mean_cooperation': opposite_data['mean_cooperation'].mean(),
            'mean_cooperation_coop_only': opposite_data['mean_cooperation_coop_only'].mean(),
            'cooperative_volume': (opposite_data['n_cooperative'].sum() / opposite_data['total_combinations'].sum()) * 100,
            'max_backfirer_fraction': opposite_data['max_backfirer_fraction'].max(),
            'mean_backfirer_fraction': opposite_data['mean_backfirer_fraction'].mean(),
            'backfirer_range': f"[{opposite_data['min_backfirer_fraction'].min():.2f}-{opposite_data['max_backfirer_fraction'].max():.2f}]",
            'stubbornness_sensitivity': opposite_data['stubbornness_sensitivity'].mean(),
            'backfirer_sensitivity': opposite_data['backfirer_sensitivity'].mean(),
            'high_polarization_percent': opposite_data['high_polarization_percent'].mean(),
            'mean_polarization': opposite_data['mean_polarization'].mean(),
            'stubbornness_range': f"[{opposite_data['min_stubbornness_coop'].min():.2f}-{opposite_data['max_stubbornness_coop'].max():.2f}]"
        }

    if len(similar_data) > 0:
        directed_comparison['similar'] = {
            'mean_cooperation': similar_data['mean_cooperation'].mean(),
            'mean_cooperation_coop_only': similar_data['mean_cooperation_coop_only'].mean(),
            'cooperative_volume': (similar_data['n_cooperative'].sum() / similar_data['total_combinations'].sum()) * 100,
            'max_backfirer_fraction': similar_data['max_backfirer_fraction'].max(),
            'mean_backfirer_fraction': similar_data['mean_backfirer_fraction'].mean(),
            'backfirer_range': f"[{similar_data['min_backfirer_fraction'].min():.2f}-{similar_data['max_backfirer_fraction'].max():.2f}]",
            'stubbornness_sensitivity': similar_data['stubbornness_sensitivity'].mean(),
            'backfirer_sensitivity': similar_data['backfirer_sensitivity'].mean(),
            'high_polarization_percent': similar_data['high_polarization_percent'].mean(),
            'mean_polarization': similar_data['mean_polarization'].mean(),
            'stubbornness_range': f"[{similar_data['min_stubbornness_coop'].min():.2f}-{similar_data['max_stubbornness_coop'].max():.2f}]"
        }

    if len(wtf_data) > 0:
        directed_comparison['wtf'] = {
            'mean_cooperation': wtf_data['mean_cooperation'].mean(),
            'mean_cooperation_coop_only': wtf_data['mean_cooperation_coop_only'].mean(),
            'cooperative_volume': (wtf_data['n_cooperative'].sum() / wtf_data['total_combinations'].sum()) * 100,
            'max_backfirer_fraction': wtf_data['max_backfirer_fraction'].max(),
            'mean_backfirer_fraction': wtf_data['mean_backfirer_fraction'].mean(),
            'backfirer_range': f"[{wtf_data['min_backfirer_fraction'].min():.2f}-{wtf_data['max_backfirer_fraction'].max():.2f}]",
            'stubbornness_sensitivity': wtf_data['stubbornness_sensitivity'].mean(),
            'backfirer_sensitivity': wtf_data['backfirer_sensitivity'].mean(),
            'high_polarization_percent': wtf_data['high_polarization_percent'].mean(),
            'mean_polarization': wtf_data['mean_polarization'].mean(),
            'stubbornness_range': f"[{wtf_data['min_stubbornness_coop'].min():.2f}-{wtf_data['max_stubbornness_coop'].max():.2f}]"
        }

    if len(n2v_data) > 0:
        directed_comparison['n2v'] = {
            'mean_cooperation': n2v_data['mean_cooperation'].mean(),
            'mean_cooperation_coop_only': n2v_data['mean_cooperation_coop_only'].mean(),
            'cooperative_volume': (n2v_data['n_cooperative'].sum() / n2v_data['total_combinations'].sum()) * 100,
            'max_backfirer_fraction': n2v_data['max_backfirer_fraction'].max(),
            'mean_backfirer_fraction': n2v_data['mean_backfirer_fraction'].mean(),
            'backfirer_range': f"[{n2v_data['min_backfirer_fraction'].min():.2f}-{n2v_data['max_backfirer_fraction'].max():.2f}]",
            'stubbornness_sensitivity': n2v_data['stubbornness_sensitivity'].mean(),
            'backfirer_sensitivity': n2v_data['backfirer_sensitivity'].mean(),
            'high_polarization_percent': n2v_data['high_polarization_percent'].mean(),
            'mean_polarization': n2v_data['mean_polarization'].mean(),
            'stubbornness_range': f"[{n2v_data['min_stubbornness_coop'].min():.2f}-{n2v_data['max_stubbornness_coop'].max():.2f}]"
        }

    if len(static_data) > 0:
        directed_comparison['static'] = {
            'mean_cooperation': static_data['mean_cooperation'].mean(),
            'mean_cooperation_coop_only': static_data['mean_cooperation_coop_only'].mean(),
            'cooperative_volume': (static_data['n_cooperative'].sum() / static_data['total_combinations'].sum()) * 100,
            'max_backfirer_fraction': static_data['max_backfirer_fraction'].max(),
            'mean_backfirer_fraction': static_data['mean_backfirer_fraction'].mean(),
            'backfirer_range': f"[{static_data['min_backfirer_fraction'].min():.2f}-{static_data['max_backfirer_fraction'].max():.2f}]",
            'stubbornness_sensitivity': static_data['stubbornness_sensitivity'].mean(),
            'backfirer_sensitivity': static_data['backfirer_sensitivity'].mean(),
            'high_polarization_percent': static_data['high_polarization_percent'].mean(),
            'mean_polarization': static_data['mean_polarization'].mean(),
            'stubbornness_range': f"[{static_data['min_stubbornness_coop'].min():.2f}-{static_data['max_stubbornness_coop'].max():.2f}]"
        }

    if len(random_data) > 0:
        directed_comparison['random'] = {
            'mean_cooperation': random_data['mean_cooperation'].mean(),
            'mean_cooperation_coop_only': random_data['mean_cooperation_coop_only'].mean(),
            'cooperative_volume': (random_data['n_cooperative'].sum() / random_data['total_combinations'].sum()) * 100,
            'max_backfirer_fraction': random_data['max_backfirer_fraction'].max(),
            'mean_backfirer_fraction': random_data['mean_backfirer_fraction'].mean(),
            'backfirer_range': f"[{random_data['min_backfirer_fraction'].min():.2f}-{random_data['max_backfirer_fraction'].max():.2f}]",
            'stubbornness_sensitivity': random_data['stubbornness_sensitivity'].mean(),
            'backfirer_sensitivity': random_data['backfirer_sensitivity'].mean(),
            'high_polarization_percent': random_data['high_polarization_percent'].mean(),
            'mean_polarization': random_data['mean_polarization'].mean(),
            'stubbornness_range': f"[{random_data['min_stubbornness_coop'].min():.2f}-{random_data['max_stubbornness_coop'].max():.2f}]"
        }
    
    return directed_comparison

def calculate_subvariant_comparison_similar(metrics_df):
    """Head-to-head comparison: local(similar) vs bridge(similar)"""
    comparison_data = []
    
    for topology in TARGET_TOPOLOGIES:
        topo_data = metrics_df[metrics_df['topology'] == topology]
        
        local_similar_data = topo_data[topo_data['friendly_name'] == 'local (similar)']
        bridge_similar_data = topo_data[topo_data['friendly_name'] == 'bridge (similar)']
        
        if len(local_similar_data) > 0 and len(bridge_similar_data) > 0:
            comparison = {
                'topology': topology,
                'local_similar_mean_coop': local_similar_data['mean_cooperation'].mean(),
                'bridge_similar_mean_coop': bridge_similar_data['mean_cooperation'].mean(),
                'local_similar_coop_only_mean': local_similar_data['mean_cooperation_coop_only'].mean(),
                'bridge_similar_coop_only_mean': bridge_similar_data['mean_cooperation_coop_only'].mean(),
                
                # Backfirer tolerance and ranges
                'local_similar_max_backfirer': local_similar_data['max_backfirer_fraction'].max(),
                'bridge_similar_max_backfirer': bridge_similar_data['max_backfirer_fraction'].max(),
                'local_similar_backfirer_range': f"[{local_similar_data['min_backfirer_fraction'].min():.2f}-{local_similar_data['max_backfirer_fraction'].max():.2f}]",
                'bridge_similar_backfirer_range': f"[{bridge_similar_data['min_backfirer_fraction'].min():.2f}-{bridge_similar_data['max_backfirer_fraction'].max():.2f}]",
                
                # Sensitivity metrics
                'local_similar_stub_sensitivity': local_similar_data['stubbornness_sensitivity'].mean(),
                'bridge_similar_stub_sensitivity': bridge_similar_data['stubbornness_sensitivity'].mean(),
                'local_similar_backf_sensitivity': local_similar_data['backfirer_sensitivity'].mean(),
                'bridge_similar_backf_sensitivity': bridge_similar_data['backfirer_sensitivity'].mean(),
                
                # Volume metrics
                'local_similar_coop_volume': (local_similar_data['n_cooperative'].sum() / local_similar_data['total_combinations'].sum()) * 100,
                'bridge_similar_coop_volume': (bridge_similar_data['n_cooperative'].sum() / bridge_similar_data['total_combinations'].sum()) * 100,
                
                # Polarization metrics
                'local_similar_high_pol_percent': local_similar_data['high_polarization_percent'].mean(),
                'bridge_similar_high_pol_percent': bridge_similar_data['high_polarization_percent'].mean(),
                'local_similar_mean_polarization': local_similar_data['mean_polarization'].mean(),
                'bridge_similar_mean_polarization': bridge_similar_data['mean_polarization'].mean(),
                
                # Stubbornness ranges
                'local_similar_stub_range': f"[{local_similar_data['min_stubbornness_coop'].min():.2f}-{local_similar_data['max_stubbornness_coop'].max():.2f}]",
                'bridge_similar_stub_range': f"[{bridge_similar_data['min_stubbornness_coop'].min():.2f}-{bridge_similar_data['max_stubbornness_coop'].max():.2f}]"
            }
            comparison_data.append(comparison)
    
    return pd.DataFrame(comparison_data)

def calculate_subvariant_comparison_opposite(metrics_df):
    """Head-to-head comparison: local(opposite) vs bridge(opposite)"""
    comparison_data = []
    
    for topology in TARGET_TOPOLOGIES:
        topo_data = metrics_df[metrics_df['topology'] == topology]
        
        local_opposite_data = topo_data[topo_data['friendly_name'] == 'local (opposite)']
        bridge_opposite_data = topo_data[topo_data['friendly_name'] == 'bridge (opposite)']
        
        if len(local_opposite_data) > 0 and len(bridge_opposite_data) > 0:
            comparison = {
                'topology': topology,
                'local_opposite_mean_coop': local_opposite_data['mean_cooperation'].mean(),
                'bridge_opposite_mean_coop': bridge_opposite_data['mean_cooperation'].mean(),
                'local_opposite_coop_only_mean': local_opposite_data['mean_cooperation_coop_only'].mean(),
                'bridge_opposite_coop_only_mean': bridge_opposite_data['mean_cooperation_coop_only'].mean(),
                
                # Backfirer tolerance and ranges
                'local_opposite_max_backfirer': local_opposite_data['max_backfirer_fraction'].max(),
                'bridge_opposite_max_backfirer': bridge_opposite_data['max_backfirer_fraction'].max(),
                'local_opposite_backfirer_range': f"[{local_opposite_data['min_backfirer_fraction'].min():.2f}-{local_opposite_data['max_backfirer_fraction'].max():.2f}]",
                'bridge_opposite_backfirer_range': f"[{bridge_opposite_data['min_backfirer_fraction'].min():.2f}-{bridge_opposite_data['max_backfirer_fraction'].max():.2f}]",
                
                # Sensitivity metrics
                'local_opposite_stub_sensitivity': local_opposite_data['stubbornness_sensitivity'].mean(),
                'bridge_opposite_stub_sensitivity': bridge_opposite_data['stubbornness_sensitivity'].mean(),
                'local_opposite_backf_sensitivity': local_opposite_data['backfirer_sensitivity'].mean(),
                'bridge_opposite_backf_sensitivity': bridge_opposite_data['backfirer_sensitivity'].mean(),
                
                # Volume metrics
                'local_opposite_coop_volume': (local_opposite_data['n_cooperative'].sum() / local_opposite_data['total_combinations'].sum()) * 100,
                'bridge_opposite_coop_volume': (bridge_opposite_data['n_cooperative'].sum() / bridge_opposite_data['total_combinations'].sum()) * 100,
                
                # Polarization metrics
                'local_opposite_high_pol_percent': local_opposite_data['high_polarization_percent'].mean(),
                'bridge_opposite_high_pol_percent': bridge_opposite_data['high_polarization_percent'].mean(),
                'local_opposite_mean_polarization': local_opposite_data['mean_polarization'].mean(),
                'bridge_opposite_mean_polarization': bridge_opposite_data['mean_polarization'].mean(),
                
                # Stubbornness ranges
                'local_opposite_stub_range': f"[{local_opposite_data['min_stubbornness_coop'].min():.2f}-{local_opposite_data['max_stubbornness_coop'].max():.2f}]",
                'bridge_opposite_stub_range': f"[{bridge_opposite_data['min_stubbornness_coop'].min():.2f}-{bridge_opposite_data['max_stubbornness_coop'].max():.2f}]"
            }
            comparison_data.append(comparison)
    
    return pd.DataFrame(comparison_data)

def calculate_topology_summary(metrics_df, exclude_wtf=True):
    """Calculate topology-specific summary statistics with option to include/exclude WTF"""
    if exclude_wtf:
        valid_metrics = metrics_df[(metrics_df['max_backfirer_fraction'] > 0) & (~metrics_df['scenario'].str.contains('wtf'))].copy()
    else:
        valid_metrics = metrics_df[metrics_df['max_backfirer_fraction'] > 0].copy()
    
    # Calculate proper cooperative volume percentage for each topology
    topology_volumes = []
    for topology in valid_metrics['topology'].unique():
        topo_data = valid_metrics[valid_metrics['topology'] == topology]
        coop_volume_percent = (topo_data['n_cooperative'].sum() / topo_data['total_combinations'].sum()) * 100
        topology_volumes.append({'topology': topology, 'cooperative_volume_percent': coop_volume_percent})
    
    volume_df = pd.DataFrame(topology_volumes).set_index('topology')
    
    summary = valid_metrics.groupby('topology').agg({
        'mean_cooperation': 'mean',
        'mean_cooperation_coop_only': 'mean', 
        'max_backfirer_fraction': ['mean', 'max'],
        'mean_backfirer_fraction': 'mean',
        'stubbornness_sensitivity': 'mean',
        'backfirer_sensitivity': 'mean',
        'high_polarization_percent': 'mean',
        'mean_polarization': 'mean'
    }).round(3)
    
    # Add the correctly calculated cooperative volume
    summary['cooperative_volume_percent'] = volume_df['cooperative_volume_percent'].round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
    
    return summary

def save_comprehensive_analysis(metrics_df, variant_comparison, directed_comparison, subvariant_similar_comparison, subvariant_opposite_comparison, topology_summary, topology_summary_with_wtf, output_dir='../../Output/Stats/stubborness_backfirer'):
    """Save all analyses to separate files"""
    today = date.today().strftime("%Y%m%d")
    
    # Round numerical columns
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    metrics_df[numeric_cols] = metrics_df[numeric_cols].round(3)
    
    # Save main metrics
    main_path = os.path.join(output_dir, f'heatmap_metrics_detailed_{today}.csv')
    metrics_df.to_csv(main_path, index=False)
    
    # Save variant comparison
    comparison_path = os.path.join(output_dir, f'variant_comparison_{today}.csv')
    variant_comparison.round(3).to_csv(comparison_path, index=False)
    
    # Save subvariant comparisons
    subvariant_similar_path = os.path.join(output_dir, f'subvariant_similar_comparison_{today}.csv')
    subvariant_similar_comparison.round(3).to_csv(subvariant_similar_path, index=False)
    
    subvariant_opposite_path = os.path.join(output_dir, f'subvariant_opposite_comparison_{today}.csv')
    subvariant_opposite_comparison.round(3).to_csv(subvariant_opposite_path, index=False)
    
    # Save topology summaries
    summary_path = os.path.join(output_dir, f'topology_summary_{today}.csv')
    topology_summary.to_csv(summary_path)
    
    summary_wtf_path = os.path.join(output_dir, f'topology_summary_with_wtf_{today}.csv')
    topology_summary_with_wtf.to_csv(summary_wtf_path)
    
    # Create comprehensive summary file
    comprehensive_path = os.path.join(output_dir, f'comprehensive_analysis_{today}.csv')
    with open(comprehensive_path, 'w') as f:
        f.write("=== DETAILED METRICS ===\n")
        metrics_df.to_csv(f, index=False)
        f.write('\n\n=== VARIANT COMPARISON (OPPOSITE vs SIMILAR vs WTF vs N2V) ===\n')
        variant_comparison.round(3).to_csv(f, index=False)
        f.write('\n\n=== SUBVARIANT COMPARISON: LOCAL(SIMILAR) vs BRIDGE(SIMILAR) ===\n')
        subvariant_similar_comparison.round(3).to_csv(f, index=False)
        f.write('\n\n=== SUBVARIANT COMPARISON: LOCAL(OPPOSITE) vs BRIDGE(OPPOSITE) ===\n')
        subvariant_opposite_comparison.round(3).to_csv(f, index=False)
        f.write('\n\n=== TOPOLOGY SUMMARY (WTF EXCLUDED) ===\n')
        topology_summary.to_csv(f)
        f.write('\n\n=== TOPOLOGY SUMMARY (WTF INCLUDED) ===\n')
        topology_summary_with_wtf.to_csv(f)
        
        # Add key statistics for paper
        f.write('\n\n=== KEY STATISTICS FOR PAPER ===\n')
        
        # Sensitivity comparison (note: higher values = more sensitive to parameter changes)
        f.write("SENSITIVITY ANALYSIS (Sobol main effects - higher = more sensitive):\n")
        for _, row in variant_comparison.iterrows():
            topo = row['topology']
            f.write(f"\n{topo} Network:\n")
            f.write(f"  Opposite variants - Stubbornness sensitivity: {row['opposite_stub_sensitivity']:.3f}, Backfirer sensitivity: {row['opposite_backf_sensitivity']:.3f}\n")
            f.write(f"  Similar variants - Stubbornness sensitivity: {row['similar_stub_sensitivity']:.3f}, Backfirer sensitivity: {row['similar_backf_sensitivity']:.3f}\n")
            f.write(f"  Backfirer ranges - Opposite: {row['opposite_backfirer_range']}, Similar: {row['similar_backfirer_range']}\n")
            f.write(f"  Cooperation volumes - Opposite: {row['opposite_coop_volume']:.1f}%, Similar: {row['similar_coop_volume']:.1f}%\n")
        
        # Max backfirer fractions by topology
        f.write("\nMAX BACKFIRER FRACTIONS BY TOPOLOGY:\n")
        max_backfirer_by_topo = metrics_df.groupby('topology')['max_backfirer_fraction'].max()
        f.write(f"{max_backfirer_by_topo.round(3).to_string()}\n\n")
        
        # Best performing scenario per topology
        best_scenarios = metrics_df.loc[metrics_df.groupby('topology')['max_backfirer_fraction'].idxmax()]
        f.write("BEST PERFORMING SCENARIOS (highest backfirer tolerance):\n")
        f.write(best_scenarios[['topology', 'friendly_name', 'max_backfirer_fraction']].to_string(index=False))
        f.write('\n\n')
        
        # Overall variant performance
        f.write("OVERALL VARIANT PERFORMANCE:\n")
        overall_opposite = metrics_df[metrics_df['variant_type'] == 'opposite']['mean_cooperation'].mean()
        overall_similar = metrics_df[metrics_df['variant_type'] == 'similar']['mean_cooperation'].mean()
        f.write(f"Mean cooperation - Opposite variants: {overall_opposite:.3f}, Similar variants: {overall_similar:.3f}\n")
        
        # Cooperation when limiting to cooperative states only
        coop_only_opposite = metrics_df[metrics_df['variant_type'] == 'opposite']['mean_cooperation_coop_only'].mean()
        coop_only_similar = metrics_df[metrics_df['variant_type'] == 'similar']['mean_cooperation_coop_only'].mean()
        f.write(f"Mean cooperation (coop states only) - Opposite: {coop_only_opposite:.3f}, Similar: {coop_only_similar:.3f}\n")
        
        # High polarization likelihood  
        high_pol_opposite = metrics_df[metrics_df['variant_type'] == 'opposite']['high_polarization_percent'].mean()
        high_pol_similar = metrics_df[metrics_df['variant_type'] == 'similar']['high_polarization_percent'].mean()
        f.write(f"High polarization likelihood - Opposite: {high_pol_opposite:.1f}%, Similar: {high_pol_similar:.1f}%\n")
        
        # Mean polarization levels
        mean_pol_opposite = metrics_df[metrics_df['variant_type'] == 'opposite']['mean_polarization'].mean()
        mean_pol_similar = metrics_df[metrics_df['variant_type'] == 'similar']['mean_polarization'].mean()
        f.write(f"Mean polarization levels - Opposite: {mean_pol_opposite:.3f}, Similar: {mean_pol_similar:.3f}\n")
        
        # Add comprehensive overall variant comparison
        f.write('\n\n=== COMPREHENSIVE OVERALL VARIANT COMPARISON (ALL TOPOLOGIES) ===\n')
        
        # Calculate overall variant metrics across all topologies
        opposite_all = metrics_df[metrics_df['variant_type'] == 'opposite']
        similar_all = metrics_df[metrics_df['variant_type'] == 'similar']
        wtf_all = metrics_df[metrics_df['friendly_name'] == 'wtf']
        n2v_all = metrics_df[metrics_df['friendly_name'] == 'node2vec']
        static_all = metrics_df[metrics_df['friendly_name'] == 'static']
        random_all = metrics_df[metrics_df['friendly_name'] == 'random']
        
        overall_comparison = {
            'opposite': {
                'mean_cooperation': opposite_all['mean_cooperation'].mean(),
                'mean_cooperation_coop_only': opposite_all['mean_cooperation_coop_only'].mean(),
                'cooperative_volume': (opposite_all['n_cooperative'].sum() / opposite_all['total_combinations'].sum()) * 100,
                'max_backfirer_fraction': opposite_all['max_backfirer_fraction'].max(),
                'mean_backfirer_fraction': opposite_all['mean_backfirer_fraction'].mean(),
                'backfirer_range': f"[{opposite_all['min_backfirer_fraction'].min():.2f}-{opposite_all['max_backfirer_fraction'].max():.2f}]",
                'stubbornness_sensitivity': opposite_all['stubbornness_sensitivity'].mean(),
                'backfirer_sensitivity': opposite_all['backfirer_sensitivity'].mean(),
                'high_polarization_percent': opposite_all['high_polarization_percent'].mean(),
                'mean_polarization': opposite_all['mean_polarization'].mean(),
                'stubbornness_range': f"[{opposite_all['min_stubbornness_coop'].min():.2f}-{opposite_all['max_stubbornness_coop'].max():.2f}]"
            },
            'similar': {
                'mean_cooperation': similar_all['mean_cooperation'].mean(),
                'mean_cooperation_coop_only': similar_all['mean_cooperation_coop_only'].mean(),
                'cooperative_volume': (similar_all['n_cooperative'].sum() / similar_all['total_combinations'].sum()) * 100,
                'max_backfirer_fraction': similar_all['max_backfirer_fraction'].max(),
                'mean_backfirer_fraction': similar_all['mean_backfirer_fraction'].mean(),
                'backfirer_range': f"[{similar_all['min_backfirer_fraction'].min():.2f}-{similar_all['max_backfirer_fraction'].max():.2f}]",
                'stubbornness_sensitivity': similar_all['stubbornness_sensitivity'].mean(),
                'backfirer_sensitivity': similar_all['backfirer_sensitivity'].mean(),
                'high_polarization_percent': similar_all['high_polarization_percent'].mean(),
                'mean_polarization': similar_all['mean_polarization'].mean(),
                'stubbornness_range': f"[{similar_all['min_stubbornness_coop'].min():.2f}-{similar_all['max_stubbornness_coop'].max():.2f}]"
            }
        }
        
        # Add all other algorithms if they exist
        if len(wtf_all) > 0:
            overall_comparison['wtf'] = {
                'mean_cooperation': wtf_all['mean_cooperation'].mean(),
                'mean_cooperation_coop_only': wtf_all['mean_cooperation_coop_only'].mean(),
                'cooperative_volume': (wtf_all['n_cooperative'].sum() / wtf_all['total_combinations'].sum()) * 100,
                'max_backfirer_fraction': wtf_all['max_backfirer_fraction'].max(),
                'mean_backfirer_fraction': wtf_all['mean_backfirer_fraction'].mean(),
                'backfirer_range': f"[{wtf_all['min_backfirer_fraction'].min():.2f}-{wtf_all['max_backfirer_fraction'].max():.2f}]",
                'stubbornness_sensitivity': wtf_all['stubbornness_sensitivity'].mean(),
                'backfirer_sensitivity': wtf_all['backfirer_sensitivity'].mean(),
                'high_polarization_percent': wtf_all['high_polarization_percent'].mean(),
                'mean_polarization': wtf_all['mean_polarization'].mean(),
                'stubbornness_range': f"[{wtf_all['min_stubbornness_coop'].min():.2f}-{wtf_all['max_stubbornness_coop'].max():.2f}]"
            }
        
        if len(n2v_all) > 0:
            overall_comparison['n2v'] = {
                'mean_cooperation': n2v_all['mean_cooperation'].mean(),
                'mean_cooperation_coop_only': n2v_all['mean_cooperation_coop_only'].mean(),
                'cooperative_volume': (n2v_all['n_cooperative'].sum() / n2v_all['total_combinations'].sum()) * 100,
                'max_backfirer_fraction': n2v_all['max_backfirer_fraction'].max(),
                'mean_backfirer_fraction': n2v_all['mean_backfirer_fraction'].mean(),
                'backfirer_range': f"[{n2v_all['min_backfirer_fraction'].min():.2f}-{n2v_all['max_backfirer_fraction'].max():.2f}]",
                'stubbornness_sensitivity': n2v_all['stubbornness_sensitivity'].mean(),
                'backfirer_sensitivity': n2v_all['backfirer_sensitivity'].mean(),
                'high_polarization_percent': n2v_all['high_polarization_percent'].mean(),
                'mean_polarization': n2v_all['mean_polarization'].mean(),
                'stubbornness_range': f"[{n2v_all['min_stubbornness_coop'].min():.2f}-{n2v_all['max_stubbornness_coop'].max():.2f}]"
            }
        
        if len(static_all) > 0:
            overall_comparison['static'] = {
                'mean_cooperation': static_all['mean_cooperation'].mean(),
                'mean_cooperation_coop_only': static_all['mean_cooperation_coop_only'].mean(),
                'cooperative_volume': (static_all['n_cooperative'].sum() / static_all['total_combinations'].sum()) * 100,
                'max_backfirer_fraction': static_all['max_backfirer_fraction'].max(),
                'mean_backfirer_fraction': static_all['mean_backfirer_fraction'].mean(),
                'backfirer_range': f"[{static_all['min_backfirer_fraction'].min():.2f}-{static_all['max_backfirer_fraction'].max():.2f}]",
                'stubbornness_sensitivity': static_all['stubbornness_sensitivity'].mean(),
                'backfirer_sensitivity': static_all['backfirer_sensitivity'].mean(),
                'high_polarization_percent': static_all['high_polarization_percent'].mean(),
                'mean_polarization': static_all['mean_polarization'].mean(),
                'stubbornness_range': f"[{static_all['min_stubbornness_coop'].min():.2f}-{static_all['max_stubbornness_coop'].max():.2f}]"
            }
        
        if len(random_all) > 0:
            overall_comparison['random'] = {
                'mean_cooperation': random_all['mean_cooperation'].mean(),
                'mean_cooperation_coop_only': random_all['mean_cooperation_coop_only'].mean(),
                'cooperative_volume': (random_all['n_cooperative'].sum() / random_all['total_combinations'].sum()) * 100,
                'max_backfirer_fraction': random_all['max_backfirer_fraction'].max(),
                'mean_backfirer_fraction': random_all['mean_backfirer_fraction'].mean(),
                'backfirer_range': f"[{random_all['min_backfirer_fraction'].min():.2f}-{random_all['max_backfirer_fraction'].max():.2f}]",
                'stubbornness_sensitivity': random_all['stubbornness_sensitivity'].mean(),
                'backfirer_sensitivity': random_all['backfirer_sensitivity'].mean(),
                'high_polarization_percent': random_all['high_polarization_percent'].mean(),
                'mean_polarization': random_all['mean_polarization'].mean(),
                'stubbornness_range': f"[{random_all['min_stubbornness_coop'].min():.2f}-{random_all['max_stubbornness_coop'].max():.2f}]"
            }
        
        # Create a proper DataFrame for the overall comparison and save as CSV
        overall_comparison_rows = []
        
        metrics_to_write = [
            ('Mean Cooperation', 'mean_cooperation'),
            ('Mean Coop (Coop Only)', 'mean_cooperation_coop_only'),
            ('Cooperative Volume %', 'cooperative_volume'),
            ('Max Backfirer Fraction', 'max_backfirer_fraction'),
            ('Mean Backfirer Fraction', 'mean_backfirer_fraction'),
            ('Backfirer Range', 'backfirer_range'),
            ('Stubbornness Sensitivity', 'stubbornness_sensitivity'),
            ('Backfirer Sensitivity', 'backfirer_sensitivity'),
            ('High Polarization %', 'high_polarization_percent'),
            ('Mean Polarization', 'mean_polarization'),
            ('Stubbornness Range', 'stubbornness_range')
        ]
        
        for metric_name, metric_key in metrics_to_write:
            row = {
                'Metric': metric_name,
                'Opposite': overall_comparison['opposite'][metric_key],
                'Similar': overall_comparison['similar'][metric_key]
            }
            
            if len(wtf_all) > 0:
                row['WTF'] = overall_comparison['wtf'][metric_key]
            
            if len(n2v_all) > 0:
                row['Node2Vec'] = overall_comparison['n2v'][metric_key]
            
            if len(static_all) > 0:
                row['Static'] = overall_comparison['static'][metric_key]
            
            if len(random_all) > 0:
                row['Random'] = overall_comparison['random'][metric_key]
                
            overall_comparison_rows.append(row)
        
        # Convert to DataFrame and write as CSV
        overall_df = pd.DataFrame(overall_comparison_rows)
        overall_df.to_csv(f, index=False)
        
        # Add directed networks head-to-head comparison
        f.write('\n\n=== DIRECTED NETWORKS HEAD-TO-HEAD COMPARISON (DPAH, Twitter only) ===\n')
        f.write('Fair comparison on directed networks where WTF is applicable\n\n')
        
        directed_comparison_rows = []
        metrics_to_write_directed = [
            ('Mean Cooperation', 'mean_cooperation'),
            ('Mean Coop (Coop Only)', 'mean_cooperation_coop_only'),
            ('Cooperative Volume %', 'cooperative_volume'),
            ('Max Backfirer Fraction', 'max_backfirer_fraction'),
            ('Mean Backfirer Fraction', 'mean_backfirer_fraction'),
            ('Backfirer Range', 'backfirer_range'),
            ('Stubbornness Sensitivity', 'stubbornness_sensitivity'),
            ('Backfirer Sensitivity', 'backfirer_sensitivity'),
            ('High Polarization %', 'high_polarization_percent'),
            ('Mean Polarization', 'mean_polarization'),
            ('Stubbornness Range', 'stubbornness_range')
        ]
        
        for metric_name, metric_key in metrics_to_write_directed:
            row = {'Metric': metric_name}

            # Only add algorithms that have data
            if 'opposite' in directed_comparison:
                row['Opposite'] = directed_comparison['opposite'][metric_key]
            if 'similar' in directed_comparison:
                row['Similar'] = directed_comparison['similar'][metric_key]
            if 'wtf' in directed_comparison:
                row['WTF'] = directed_comparison['wtf'][metric_key]
            if 'n2v' in directed_comparison:
                row['Node2Vec'] = directed_comparison['n2v'][metric_key]
            if 'static' in directed_comparison:
                row['Static'] = directed_comparison['static'][metric_key]
            if 'random' in directed_comparison:
                row['Random'] = directed_comparison['random'][metric_key]

            directed_comparison_rows.append(row)
        
        directed_df = pd.DataFrame(directed_comparison_rows)
        directed_df.to_csv(f, index=False)
        
        # Add subvariant analysis summaries
        f.write('\n\n=== SUBVARIANT ANALYSIS: SIMILAR VARIANTS HEAD-TO-HEAD ===\n')
        f.write('Comparing local(similar) vs bridge(similar) algorithms:\n\n')
        
        for _, row in subvariant_similar_comparison.iterrows():
            topo = row['topology']
            f.write(f"{topo} Network:\n")
            f.write(f"  Local(similar): Mean coop = {row['local_similar_mean_coop']:.3f}, Max backfirer = {row['local_similar_max_backfirer']:.3f}, Coop volume = {row['local_similar_coop_volume']:.1f}%\n")
            f.write(f"  Bridge(similar): Mean coop = {row['bridge_similar_mean_coop']:.3f}, Max backfirer = {row['bridge_similar_max_backfirer']:.3f}, Coop volume = {row['bridge_similar_coop_volume']:.1f}%\n")
            f.write(f"  Winner: {'Local(similar)' if row['local_similar_max_backfirer'] > row['bridge_similar_max_backfirer'] else 'Bridge(similar)'} (higher backfirer tolerance)\n\n")
        
        f.write('\n=== SUBVARIANT ANALYSIS: OPPOSITE VARIANTS HEAD-TO-HEAD ===\n')
        f.write('Comparing local(opposite) vs bridge(opposite) algorithms:\n\n')
        
        for _, row in subvariant_opposite_comparison.iterrows():
            topo = row['topology']
            f.write(f"{topo} Network:\n")
            f.write(f"  Local(opposite): Mean coop = {row['local_opposite_mean_coop']:.3f}, Max backfirer = {row['local_opposite_max_backfirer']:.3f}, Coop volume = {row['local_opposite_coop_volume']:.1f}%\n")
            f.write(f"  Bridge(opposite): Mean coop = {row['bridge_opposite_mean_coop']:.3f}, Max backfirer = {row['bridge_opposite_max_backfirer']:.3f}, Coop volume = {row['bridge_opposite_coop_volume']:.1f}%\n")
            f.write(f"  Winner: {'Local(opposite)' if row['local_opposite_max_backfirer'] > row['bridge_opposite_max_backfirer'] else 'Bridge(opposite)'} (higher backfirer tolerance)\n\n")
    
    return comprehensive_path, main_path, comparison_path, subvariant_similar_path, subvariant_opposite_path, summary_path, summary_wtf_path

def main():
    """Main execution function"""
    data_path = os.environ.get("HEATMAP_CSV")
    if data_path:
        print(f"Using HEATMAP_CSV override: {data_path}")
    else:
        file_list = [f for f in os.listdir("../../Output") if f.endswith(".csv") and "heatmap" in f]

        if not file_list:
            print("No suitable files found in the Output directory.")
            return

        for i, file in enumerate(file_list):
            print(f"{i}: {file}")

        file_index = 0  # Auto-select first file
        print(f"Auto-selecting file {file_index}: {file_list[file_index]}")
        data_path = os.path.join("../../Output", file_list[file_index])
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    df.loc[df['mode'].isin(['wtf', 'node2vec']), 'rewiring'] = 'empirical'
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    df['scenario'] = df['rewiring'] + ' ' + df['mode']
    
    print(f"\nAnalyzing {TARGET_TOPOLOGIES} networks...")
    print("Calculating comprehensive metrics...")
    
    # Calculate all metrics
    metrics_df = calculate_metrics(df)
    variant_comparison = calculate_variant_comparison(metrics_df)
    directed_comparison = calculate_directed_networks_comparison(metrics_df)
    subvariant_similar_comparison = calculate_subvariant_comparison_similar(metrics_df)
    subvariant_opposite_comparison = calculate_subvariant_comparison_opposite(metrics_df)
    topology_summary = calculate_topology_summary(metrics_df, exclude_wtf=True)
    topology_summary_with_wtf = calculate_topology_summary(metrics_df, exclude_wtf=False)
    
    # Save results
    paths = save_comprehensive_analysis(metrics_df, variant_comparison, directed_comparison, subvariant_similar_comparison, subvariant_opposite_comparison, topology_summary, topology_summary_with_wtf)
    
    print(f"\nAnalysis complete! Files saved:")
    for i, path in enumerate(paths):
        print(f"{i+1}. {path}")
    
    # Print key findings
    print(f"\n=== KEY FINDINGS ===")
    print(f"Total cooperative combinations analyzed: {metrics_df['n_cooperative'].sum()}")
    print(f"Topologies: {', '.join(TARGET_TOPOLOGIES)}")
    
    # Print max backfirer fractions
    print(f"\nMax backfirer fractions by topology:")
    max_by_topo = metrics_df.groupby('topology')['max_backfirer_fraction'].max()
    for topo, max_val in max_by_topo.items():
        print(f"  {topo}: {max_val:.3f}")
    
    return metrics_df, variant_comparison, directed_comparison, subvariant_similar_comparison, subvariant_opposite_comparison, topology_summary, topology_summary_with_wtf

if __name__ == "__main__":
    main()