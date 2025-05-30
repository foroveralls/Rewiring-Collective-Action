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
        
        # Stubbornness robustness analysis
        stub_coop_by_level = []
        if 'stubbornness' in group.columns:
            for stub_level in group['stubbornness'].unique():
                stub_data = group[group['stubbornness'] == stub_level]
                stub_coop_ratio = (stub_data['state'] > 0).mean()
                stub_coop_by_level.append(stub_coop_ratio)
        
        stub_robustness = np.std(stub_coop_by_level) if len(stub_coop_by_level) > 1 else 0.0
        
        # Backfirer robustness analysis  
        backf_coop_by_level = []
        if 'polarisingNode_f' in group.columns:
            for backf_level in group['polarisingNode_f'].unique():
                backf_data = group[group['polarisingNode_f'] == backf_level]
                backf_coop_ratio = (backf_data['state'] > 0).mean()
                backf_coop_by_level.append(backf_coop_ratio)
        
        backf_robustness = np.std(backf_coop_by_level) if len(backf_coop_by_level) > 1 else 0.0
        
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
            
            # Robustness metrics (lower = more robust)
            'stubbornness_robustness': stub_robustness,  # Std of coop ratios across stubbornness levels
            'backfirer_robustness': backf_robustness,    # Std of coop ratios across backfirer levels
            
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
    """Compare opposite vs similar variants"""
    comparison_data = []
    
    for topology in TARGET_TOPOLOGIES:
        topo_data = metrics_df[metrics_df['topology'] == topology]
        
        opposite_data = topo_data[topo_data['variant_type'] == 'opposite']
        similar_data = topo_data[topo_data['variant_type'] == 'similar']
        
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
                
                # Robustness metrics (lower = more robust)
                'opposite_stub_robustness': opposite_data['stubbornness_robustness'].mean(),
                'similar_stub_robustness': similar_data['stubbornness_robustness'].mean(),
                'opposite_backf_robustness': opposite_data['backfirer_robustness'].mean(), 
                'similar_backf_robustness': similar_data['backfirer_robustness'].mean(),
                
                # Volume metrics
                'opposite_coop_volume': opposite_data['cooperative_volume_percent'].sum(),
                'similar_coop_volume': similar_data['cooperative_volume_percent'].sum(),
                
                # Polarization metrics
                'opposite_high_pol_percent': opposite_data['high_polarization_percent'].mean(),
                'similar_high_pol_percent': similar_data['high_polarization_percent'].mean(),
                'opposite_mean_polarization': opposite_data['mean_polarization'].mean(),
                'similar_mean_polarization': similar_data['mean_polarization'].mean(),
                
                # Stubbornness ranges for cooperative states
                'opposite_stub_range': f"[{opposite_data['min_stubbornness_coop'].min():.2f}-{opposite_data['max_stubbornness_coop'].max():.2f}]",
                'similar_stub_range': f"[{similar_data['min_stubbornness_coop'].min():.2f}-{similar_data['max_stubbornness_coop'].max():.2f}]"
            }
            comparison_data.append(comparison)
    
    return pd.DataFrame(comparison_data)

def calculate_topology_summary(metrics_df, exclude_wtf=True):
    """Calculate topology-specific summary statistics with option to include/exclude WTF"""
    if exclude_wtf:
        valid_metrics = metrics_df[(metrics_df['max_backfirer_fraction'] > 0) & (~metrics_df['scenario'].str.contains('wtf'))].copy()
    else:
        valid_metrics = metrics_df[metrics_df['max_backfirer_fraction'] > 0].copy()
    
    summary = valid_metrics.groupby('topology').agg({
        'mean_cooperation': 'mean',
        'mean_cooperation_coop_only': 'mean', 
        'cooperative_volume_percent': 'sum',
        'max_backfirer_fraction': ['mean', 'max'],
        'mean_backfirer_fraction': 'mean',
        'stubbornness_robustness': 'mean',
        'backfirer_robustness': 'mean',
        'high_polarization_percent': 'mean',
        'mean_polarization': 'mean'
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
    
    return summary

def save_comprehensive_analysis(metrics_df, variant_comparison, topology_summary, topology_summary_with_wtf, output_dir='../../Output/Stats/stubborness_backfirer'):
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
        f.write('\n\n=== VARIANT COMPARISON (OPPOSITE vs SIMILAR) ===\n')
        variant_comparison.round(3).to_csv(f, index=False)
        f.write('\n\n=== TOPOLOGY SUMMARY (WTF EXCLUDED) ===\n')
        topology_summary.to_csv(f)
        f.write('\n\n=== TOPOLOGY SUMMARY (WTF INCLUDED) ===\n')
        topology_summary_with_wtf.to_csv(f)
        
        # Add key statistics for paper
        f.write('\n\n=== KEY STATISTICS FOR PAPER ===\n')
        
        # Robustness comparison
        f.write("ROBUSTNESS ANALYSIS:\n")
        for _, row in variant_comparison.iterrows():
            topo = row['topology']
            f.write(f"\n{topo} Network:\n")
            f.write(f"  Opposite variants - Stubbornness robustness: {row['opposite_stub_robustness']:.3f}, Backfirer robustness: {row['opposite_backf_robustness']:.3f}\n")
            f.write(f"  Similar variants - Stubbornness robustness: {row['similar_stub_robustness']:.3f}, Backfirer robustness: {row['similar_backf_robustness']:.3f}\n")
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
    
    return comprehensive_path, main_path, comparison_path, summary_path, summary_wtf_path

def main():
    """Main execution function"""
    file_list = [f for f in os.listdir("../../Output") if f.endswith(".csv") and "heatmap" in f]
    
    if not file_list:
        print("No suitable files found in the Output directory.")
        return
    
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    file_index = int(input("Enter the index of the file you want to analyze: "))
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
    topology_summary = calculate_topology_summary(metrics_df, exclude_wtf=True)
    topology_summary_with_wtf = calculate_topology_summary(metrics_df, exclude_wtf=False)
    
    # Save results
    paths = save_comprehensive_analysis(metrics_df, variant_comparison, topology_summary, topology_summary_with_wtf)
    
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
    
    return metrics_df, variant_comparison, topology_summary, topology_summary_with_wtf

if __name__ == "__main__":
    main()