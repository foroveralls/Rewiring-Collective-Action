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

# Define stubbornness regimes for regime-based analysis
STUBBORNNESS_REGIMES = {
    'low': (0.0, 0.4),      # Low stubbornness - limited cooperation possible
    'medium': (0.4, 0.7),   # Medium stubbornness - most realistic/practical range
    'high': (0.7, 1.0)      # High stubbornness - extreme regime where WTF excels
}

# Practical importance weights for different regimes (sum to 1.0)
REGIME_WEIGHTS = {
    'low': 0.2,     # Less important - limited cooperation anyway
    'medium': 0.6,  # Most important - realistic parameter range
    'high': 0.2     # Less important - extreme/unrealistic range
}

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

def get_stubbornness_regime(stubbornness_value):
    """Classify stubbornness value into regime"""
    for regime, (min_val, max_val) in STUBBORNNESS_REGIMES.items():
        if min_val <= stubbornness_value < max_val:
            return regime
    # Handle edge case for stubbornness = 1.0
    if stubbornness_value >= 0.7:
        return 'high'
    return 'low'

def calculate_regime_based_metrics(df):
    """Calculate performance metrics stratified by stubbornness regime"""
    # Filter to target topologies
    df = df[df['topology'].isin(TARGET_TOPOLOGIES)].copy()
    
    # Add regime classification
    df['stubbornness_regime'] = df['stubbornness'].apply(get_stubbornness_regime)
    
    all_metrics = []
    
    for (topology, scenario, regime), group in df.groupby(['topology', 'scenario', 'stubbornness_regime']):
        friendly_name = get_friendly_name(scenario)
        variant_type = calculate_variant_type(friendly_name)
        
        # Basic metrics for this regime
        total_combinations = len(group)
        cooperative_mask = group['state'] > 0
        cooperative_data = group[cooperative_mask]
        n_cooperative = len(cooperative_data)
        
        metrics = {
            'topology': topology,
            'scenario': scenario,
            'friendly_name': friendly_name,
            'variant_type': variant_type,
            'stubbornness_regime': regime,
            
            # Performance in this regime
            'mean_cooperation': group['state'].mean(),
            'median_cooperation': group['state'].median(),
            'cooperative_ratio': n_cooperative / total_combinations if total_combinations > 0 else 0,
            'cooperative_volume_percent': (n_cooperative / total_combinations) * 100 if total_combinations > 0 else 0,
            
            # Cooperative-only metrics
            'mean_cooperation_coop_only': cooperative_data['state'].mean() if n_cooperative > 0 else np.nan,
            
            # Backfirer metrics for cooperative states
            'max_backfirer_fraction': np.max(cooperative_data['polarisingNode_f']) if n_cooperative > 0 else 0.0,
            'mean_backfirer_fraction': np.mean(cooperative_data['polarisingNode_f']) if n_cooperative > 0 else 0.0,
            
            # Polarization metrics
            'mean_polarization': group['state_std'].mean(),
            'high_polarization_percent': (len(group[group['state_std'] >= 0.8]) / total_combinations) * 100 if total_combinations > 0 else 0,
            
            # Sample size
            'n_combinations': total_combinations,
            'n_cooperative': n_cooperative
        }
        
        all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)

def calculate_practical_performance(df):
    """Calculate performance focusing on the medium stubbornness regime (most realistic)"""
    # Filter to medium stubbornness regime only
    df_medium = df[(df['stubbornness'] >= 0.4) & (df['stubbornness'] < 0.7)].copy()
    df_medium = df_medium[df_medium['topology'].isin(TARGET_TOPOLOGIES)]
    
    all_metrics = []
    
    for (topology, scenario), group in df_medium.groupby(['topology', 'scenario']):
        friendly_name = get_friendly_name(scenario)
        variant_type = calculate_variant_type(friendly_name)
        
        # Calculate metrics for medium stubbornness range only
        total_combinations = len(group)
        cooperative_mask = group['state'] > 0
        cooperative_data = group[cooperative_mask]
        n_cooperative = len(cooperative_data)
        
        metrics = {
            'topology': topology,
            'scenario': scenario,
            'friendly_name': friendly_name,
            'variant_type': variant_type,
            
            # Practical performance metrics (medium stubbornness only)
            'practical_mean_cooperation': group['state'].mean(),
            'practical_cooperative_ratio': n_cooperative / total_combinations if total_combinations > 0 else 0,
            'practical_max_backfirer': np.max(cooperative_data['polarisingNode_f']) if n_cooperative > 0 else 0.0,
            'practical_mean_backfirer': np.mean(cooperative_data['polarisingNode_f']) if n_cooperative > 0 else 0.0,
            'practical_mean_polarization': group['state_std'].mean(),
            
            # Sample size for medium regime
            'practical_n_combinations': total_combinations,
            'practical_n_cooperative': n_cooperative
        }
        
        all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)

def calculate_regime_weighted_averages(regime_metrics_df):
    """Calculate weighted averages across regimes using practical importance weights"""
    
    all_weighted_metrics = []
    
    for (topology, scenario), scenario_group in regime_metrics_df.groupby(['topology', 'scenario']):
        friendly_name = get_friendly_name(scenario)
        variant_type = calculate_variant_type(friendly_name)
        
        # Initialize weighted sums
        weighted_cooperation = 0
        weighted_backfirer = 0
        weighted_polarization = 0
        weighted_coop_ratio = 0
        total_weight = 0
        
        # Calculate weighted averages across regimes
        for regime in ['low', 'medium', 'high']:
            regime_data = scenario_group[scenario_group['stubbornness_regime'] == regime]
            
            if len(regime_data) > 0:
                weight = REGIME_WEIGHTS[regime]
                regime_row = regime_data.iloc[0]  # Should be only one row per regime
                
                weighted_cooperation += weight * regime_row['mean_cooperation']
                weighted_backfirer += weight * regime_row['max_backfirer_fraction']
                weighted_polarization += weight * regime_row['mean_polarization']
                weighted_coop_ratio += weight * regime_row['cooperative_ratio']
                total_weight += weight
        
        # Normalize by total weight (handles missing regimes)
        if total_weight > 0:
            metrics = {
                'topology': topology,
                'scenario': scenario,
                'friendly_name': friendly_name,
                'variant_type': variant_type,
                
                # Regime-weighted performance metrics
                'weighted_mean_cooperation': weighted_cooperation / total_weight,
                'weighted_cooperative_ratio': weighted_coop_ratio / total_weight,
                'weighted_max_backfirer': weighted_backfirer / total_weight,
                'weighted_mean_polarization': weighted_polarization / total_weight,
                
                # Total effective weight (for quality assessment)
                'total_regime_weight': total_weight
            }
            
            all_weighted_metrics.append(metrics)
    
    return pd.DataFrame(all_weighted_metrics)

def calculate_conditional_performance(regime_metrics_df):
    """Determine which strategy performs best in each regime for each topology"""
    
    conditional_results = []
    
    for topology in TARGET_TOPOLOGIES:
        topo_data = regime_metrics_df[regime_metrics_df['topology'] == topology]
        
        regime_winners = {}
        regime_performance = {}
        
        for regime in ['low', 'medium', 'high']:
            regime_data = topo_data[topo_data['stubbornness_regime'] == regime]
            
            if len(regime_data) > 0:
                # Find best strategy by mean cooperation in this regime
                best_idx = regime_data['mean_cooperation'].idxmax()
                best_strategy = regime_data.loc[best_idx, 'friendly_name']
                best_cooperation = regime_data.loc[best_idx, 'mean_cooperation']
                best_backfirer = regime_data.loc[best_idx, 'max_backfirer_fraction']
                
                # Find second best for comparison
                sorted_data = regime_data.sort_values('mean_cooperation', ascending=False)
                second_best_cooperation = sorted_data.iloc[1]['mean_cooperation'] if len(sorted_data) > 1 else 0
                performance_gap = best_cooperation - second_best_cooperation
                
                regime_winners[f'{regime}_winner'] = best_strategy
                regime_winners[f'{regime}_cooperation'] = best_cooperation
                regime_winners[f'{regime}_backfirer'] = best_backfirer
                regime_winners[f'{regime}_gap'] = performance_gap
                
                # Store all strategies' performance in this regime for ranking
                regime_performance[regime] = sorted_data[['friendly_name', 'mean_cooperation', 'max_backfirer_fraction']].to_dict('records')
        
        result = {
            'topology': topology,
            **regime_winners,
            'regime_performance_details': regime_performance
        }
        
        conditional_results.append(result)
    
    return pd.DataFrame(conditional_results)

def calculate_cross_regime_consistency(regime_metrics_df):
    """Calculate how consistently each strategy performs across regimes"""
    
    consistency_metrics = []
    
    for (topology, scenario), scenario_group in regime_metrics_df.groupby(['topology', 'scenario']):
        friendly_name = get_friendly_name(scenario)
        variant_type = calculate_variant_type(friendly_name)
        
        # Get performance across all regimes
        regime_performances = []
        regime_backfirers = []
        
        for regime in ['low', 'medium', 'high']:
            regime_data = scenario_group[scenario_group['stubbornness_regime'] == regime]
            if len(regime_data) > 0:
                regime_performances.append(regime_data.iloc[0]['mean_cooperation'])
                regime_backfirers.append(regime_data.iloc[0]['max_backfirer_fraction'])
        
        if len(regime_performances) >= 2:
            # Calculate consistency metrics
            cooperation_std = np.std(regime_performances)
            cooperation_range = max(regime_performances) - min(regime_performances)
            backfirer_std = np.std(regime_backfirers)
            
            metrics = {
                'topology': topology,
                'scenario': scenario,
                'friendly_name': friendly_name,
                'variant_type': variant_type,
                
                # Consistency metrics (lower = more consistent)
                'cooperation_consistency_std': cooperation_std,
                'cooperation_range': cooperation_range,
                'backfirer_consistency_std': backfirer_std,
                
                # Performance by regime
                'low_regime_cooperation': regime_performances[0] if len(regime_performances) > 0 else np.nan,
                'medium_regime_cooperation': regime_performances[1] if len(regime_performances) > 1 else np.nan,
                'high_regime_cooperation': regime_performances[2] if len(regime_performances) > 2 else np.nan,
                
                'n_regimes_available': len(regime_performances)
            }
            
            consistency_metrics.append(metrics)
    
    return pd.DataFrame(consistency_metrics)

def calculate_comprehensive_algorithm_comparison(regime_metrics_df):
    """Calculate comprehensive algorithm comparison tables like the original analysis"""
    
    all_comparison_tables = {}
    
    # For each regime, create comprehensive comparison
    for regime in ['low', 'medium', 'high', 'all_regimes']:
        if regime == 'all_regimes':
            # Use all data for overall comparison
            regime_data = regime_metrics_df
        else:
            # Filter to specific regime
            regime_data = regime_metrics_df[regime_metrics_df['stubbornness_regime'] == regime]
        
        if len(regime_data) == 0:
            continue
            
        # Calculate algorithm-level aggregations
        algorithm_stats = {}
        
        for algorithm in ['local (opposite)', 'bridge (opposite)', 'local (similar)', 'bridge (similar)', 'empirical wtf', 'empirical node2vec', 'static', 'random']:
            algo_data = regime_data[regime_data['friendly_name'] == algorithm]
            
            if len(algo_data) > 0:
                # Calculate comprehensive metrics
                total_combinations = algo_data['n_combinations'].sum()
                total_cooperative = algo_data['n_cooperative'].sum()
                
                algorithm_stats[algorithm] = {
                    'mean_cooperation': algo_data['mean_cooperation'].mean(),
                    'mean_cooperation_coop_only': algo_data['mean_cooperation_coop_only'].mean(),
                    'cooperative_volume_percent': (total_cooperative / total_combinations) * 100 if total_combinations > 0 else 0,
                    'max_backfirer_fraction': algo_data['max_backfirer_fraction'].max(),
                    'mean_backfirer_fraction': algo_data['mean_backfirer_fraction'].mean(),
                    'mean_polarization': algo_data['mean_polarization'].mean(),
                    'high_polarization_percent': algo_data['high_polarization_percent'].mean(),
                    'n_topologies': len(algo_data),
                    'total_combinations': total_combinations,
                    'total_cooperative': total_cooperative
                }
        
        # Create comparison table
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
        
        for metric_name, metric_key in metrics_to_compare:
            row = {'Metric': metric_name}
            
            # Add data for each algorithm category
            opposite_algos = ['local (opposite)', 'bridge (opposite)']
            similar_algos = ['local (similar)', 'bridge (similar)']
            
            # Opposite variants aggregate
            opposite_values = [algorithm_stats[algo][metric_key] for algo in opposite_algos if algo in algorithm_stats]
            if opposite_values:
                row['Opposite'] = np.mean(opposite_values)
            
            # Similar variants aggregate  
            similar_values = [algorithm_stats[algo][metric_key] for algo in similar_algos if algo in algorithm_stats]
            if similar_values:
                row['Similar'] = np.mean(similar_values)
            
            # Individual algorithms
            individual_algos = [
                ('empirical wtf', 'WTF'),
                ('empirical node2vec', 'Node2Vec'),
                ('static', 'Static'),
                ('random', 'Random')
            ]
            
            for algo_key, algo_display in individual_algos:
                if algo_key in algorithm_stats:
                    row[algo_display] = algorithm_stats[algo_key][metric_key]
            
            comparison_rows.append(row)
        
        all_comparison_tables[regime] = pd.DataFrame(comparison_rows)
    
    return all_comparison_tables

def calculate_topology_regime_summary(regime_metrics_df):
    """Calculate topology-level performance by regime"""
    
    topology_summaries = {}
    
    for regime in ['low', 'medium', 'high']:
        regime_data = regime_metrics_df[regime_metrics_df['stubbornness_regime'] == regime]
        
        if len(regime_data) > 0:
            # Group by topology and calculate summary stats
            topo_summary = regime_data.groupby('topology').agg({
                'mean_cooperation': 'mean',
                'mean_cooperation_coop_only': 'mean',
                'cooperative_ratio': 'mean', 
                'max_backfirer_fraction': 'max',
                'mean_backfirer_fraction': 'mean',
                'mean_polarization': 'mean',
                'high_polarization_percent': 'mean'
            }).round(3)
            
            # Add cooperative volume calculation
            regime_volume = regime_data.groupby('topology').apply(
                lambda x: (x['n_cooperative'].sum() / x['n_combinations'].sum()) * 100
            ).round(3)
            
            topo_summary['cooperative_volume_percent'] = regime_volume
            
            topology_summaries[regime] = topo_summary
    
    return topology_summaries

def generate_regime_based_summary(regime_metrics_df, practical_metrics_df, weighted_metrics_df, conditional_df, consistency_df):
    """Generate comprehensive summary addressing the WTF regime-dependent performance issue"""
    
    summary_sections = []
    
    # Section 1: Overall vs Practical Performance Comparison
    summary_sections.append("=== OVERALL vs PRACTICAL PERFORMANCE COMPARISON ===")
    summary_sections.append("This analysis addresses the counter-intuitive WTF results by separating")
    summary_sections.append("overall performance (biased by extreme regimes) from practical performance")
    summary_sections.append("(focused on realistic parameter ranges).\n")
    
    # Compare overall vs practical performance for key strategies
    key_strategies = ['wtf', 'local (opposite)', 'bridge (opposite)', 'static']
    
    for topology in TARGET_TOPOLOGIES:
        summary_sections.append(f"{topology} Network:")
        
        # Get overall performance (from original weighted analysis)
        topo_weighted = weighted_metrics_df[weighted_metrics_df['topology'] == topology]
        topo_practical = practical_metrics_df[practical_metrics_df['topology'] == topology]
        
        for strategy in key_strategies:
            weighted_data = topo_weighted[topo_weighted['friendly_name'] == strategy]
            practical_data = topo_practical[topo_practical['friendly_name'] == strategy]
            
            if len(weighted_data) > 0 and len(practical_data) > 0:
                weighted_coop = weighted_data.iloc[0]['weighted_mean_cooperation']
                practical_coop = practical_data.iloc[0]['practical_mean_cooperation']
                performance_bias = weighted_coop - practical_coop
                
                summary_sections.append(f"  {strategy}:")
                summary_sections.append(f"    Regime-weighted performance: {weighted_coop:.3f}")
                summary_sections.append(f"    Practical performance (medium stubbornness): {practical_coop:.3f}")
                summary_sections.append(f"    Bias from extreme regimes: {performance_bias:+.3f}")
        
        summary_sections.append("")
    
    # Section 2: Regime-Specific Winners
    summary_sections.append("\n=== BEST STRATEGY BY REGIME ===")
    summary_sections.append("Shows which strategy performs best in each stubbornness regime:\n")
    
    for _, row in conditional_df.iterrows():
        topology = row['topology']
        summary_sections.append(f"{topology} Network:")
        
        for regime in ['low', 'medium', 'high']:
            regime_range = STUBBORNNESS_REGIMES[regime]
            winner = row.get(f'{regime}_winner', 'N/A')
            cooperation = row.get(f'{regime}_cooperation', 0)
            gap = row.get(f'{regime}_gap', 0)
            
            summary_sections.append(f"  {regime.title()} stubbornness ({regime_range[0]:.1f}-{regime_range[1]:.1f}): {winner} (cooperation={cooperation:.3f}, gap={gap:.3f})")
        
        summary_sections.append("")
    
    # Section 3: Practical Recommendations
    summary_sections.append("\n=== PRACTICAL RECOMMENDATIONS ===")
    summary_sections.append("Based on realistic parameter ranges (medium stubbornness regime):\n")
    
    # Rank strategies by practical performance
    for topology in TARGET_TOPOLOGIES:
        topo_practical = practical_metrics_df[practical_metrics_df['topology'] == topology]
        
        if len(topo_practical) > 0:
            # Sort by practical performance
            topo_practical_sorted = topo_practical.sort_values('practical_mean_cooperation', ascending=False)
            
            summary_sections.append(f"{topology} Network (ranked by practical performance):")
            
            for i, (_, row) in enumerate(topo_practical_sorted.iterrows(), 1):
                strategy = row['friendly_name']
                coop = row['practical_mean_cooperation']
                backfirer = row['practical_max_backfirer']
                
                summary_sections.append(f"  {i}. {strategy}: cooperation={coop:.3f}, max_backfirer={backfirer:.3f}")
            
            summary_sections.append("")
    
    # Section 4: Regime Consistency Analysis
    summary_sections.append("\n=== STRATEGY CONSISTENCY ACROSS REGIMES ===")
    summary_sections.append("Strategies with lower values are more consistent across parameter ranges:\n")
    
    for topology in TARGET_TOPOLOGIES:
        topo_consistency = consistency_df[consistency_df['topology'] == topology]
        
        if len(topo_consistency) > 0:
            # Sort by consistency (lower std = more consistent)
            topo_consistency_sorted = topo_consistency.sort_values('cooperation_consistency_std')
            
            summary_sections.append(f"{topology} Network (ranked by consistency):")
            
            for i, (_, row) in enumerate(topo_consistency_sorted.iterrows(), 1):
                strategy = row['friendly_name']
                std = row['cooperation_consistency_std']
                range_val = row['cooperation_range']
                
                summary_sections.append(f"  {i}. {strategy}: std={std:.3f}, range={range_val:.3f}")
            
            summary_sections.append("")
    
    return "\n".join(summary_sections)

def save_regime_based_analysis(regime_metrics_df, practical_metrics_df, weighted_metrics_df, conditional_df, consistency_df, summary_text, comprehensive_tables, topology_summaries, output_dir='../../Output/Stats/stubborness_backfirer'):
    """Save all regime-based analyses to files"""
    
    today = date.today().strftime("%Y%m%d")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # Save regime-based metrics
    regime_path = os.path.join(output_dir, f'regime_based_metrics_{today}.csv')
    regime_metrics_df.round(3).to_csv(regime_path, index=False)
    saved_files.append(regime_path)
    
    # Save practical performance metrics
    practical_path = os.path.join(output_dir, f'practical_performance_{today}.csv')
    practical_metrics_df.round(3).to_csv(practical_path, index=False)
    saved_files.append(practical_path)
    
    # Save weighted averages
    weighted_path = os.path.join(output_dir, f'regime_weighted_averages_{today}.csv')
    weighted_metrics_df.round(3).to_csv(weighted_path, index=False)
    saved_files.append(weighted_path)
    
    # Save conditional performance
    conditional_path = os.path.join(output_dir, f'conditional_performance_{today}.csv')
    conditional_df.to_csv(conditional_path, index=False)
    saved_files.append(conditional_path)
    
    # Save consistency metrics
    consistency_path = os.path.join(output_dir, f'cross_regime_consistency_{today}.csv')
    consistency_df.round(3).to_csv(consistency_path, index=False)
    saved_files.append(consistency_path)
    
    # Save comprehensive algorithm comparison tables
    for regime_name, table in comprehensive_tables.items():
        table_path = os.path.join(output_dir, f'comprehensive_algorithm_comparison_{regime_name}_{today}.csv')
        table.round(3).to_csv(table_path, index=False)
        saved_files.append(table_path)
    
    # Save topology regime summaries
    for regime_name, summary in topology_summaries.items():
        topo_path = os.path.join(output_dir, f'topology_summary_{regime_name}_regime_{today}.csv')
        summary.to_csv(topo_path)
        saved_files.append(topo_path)
    
    # Save comprehensive summary with tables included
    summary_path = os.path.join(output_dir, f'regime_based_comprehensive_analysis_{today}.txt')
    with open(summary_path, 'w') as f:
        f.write("=== REGIME-BASED COMPREHENSIVE ANALYSIS ===\n")
        f.write("This analysis addresses WTF's counter-intuitive performance by separating results into stubbornness regimes.\n\n")
        
        # Add file descriptions
        f.write("=== FILE DESCRIPTIONS ===\n")
        f.write("1. regime_based_metrics_YYYYMMDD.csv: Performance metrics broken down by (topology, algorithm, stubbornness_regime)\n")
        f.write("2. practical_performance_YYYYMMDD.csv: Performance in medium stubbornness regime only (0.4-0.7) - most realistic\n")
        f.write("3. regime_weighted_averages_YYYYMMDD.csv: Weighted averages across regimes (low=20%, medium=60%, high=20%)\n")
        f.write("4. conditional_performance_YYYYMMDD.csv: Best-performing algorithm in each regime for each topology\n")
        f.write("5. cross_regime_consistency_YYYYMMDD.csv: How consistently each algorithm performs across regimes\n")
        f.write("6. comprehensive_algorithm_comparison_[regime]_YYYYMMDD.csv: Algorithm comparison tables by regime\n")
        f.write("7. topology_summary_[regime]_regime_YYYYMMDD.csv: Topology-level performance summaries by regime\n\n")
        
        # Add comprehensive comparison tables
        for regime_name, table in comprehensive_tables.items():
            regime_title = regime_name.replace('_', ' ').title()
            if regime_name == 'all_regimes':
                regime_title = 'Overall (All Regimes Combined)'
            
            f.write(f"=== COMPREHENSIVE ALGORITHM COMPARISON - {regime_title.upper()} STUBBORNNESS ===\n")
            if regime_name != 'all_regimes':
                regime_range = STUBBORNNESS_REGIMES[regime_name]
                f.write(f"Stubbornness range: {regime_range[0]:.1f} - {regime_range[1]:.1f}\n")
            f.write("\n")
            table.round(3).to_csv(f, index=False)
            f.write('\n\n')
        
        # Add topology summaries
        for regime_name, summary in topology_summaries.items():
            regime_title = regime_name.replace('_', ' ').title() 
            regime_range = STUBBORNNESS_REGIMES[regime_name]
            f.write(f"=== TOPOLOGY SUMMARY - {regime_title.upper()} STUBBORNNESS ({regime_range[0]:.1f}-{regime_range[1]:.1f}) ===\n")
            summary.to_csv(f)
            f.write('\n\n')
        
        # Add the original summary text
        f.write("=== DETAILED ANALYSIS ===\n")
        f.write(summary_text)
        
    saved_files.append(summary_path)
    
    return saved_files

def main():
    """Main execution function for regime-based analysis"""
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
    
    print(f"\nAnalyzing {TARGET_TOPOLOGIES} networks with regime-based approach...")
    print("Calculating regime-stratified metrics...")
    
    # Calculate all regime-based metrics
    regime_metrics_df = calculate_regime_based_metrics(df)
    practical_metrics_df = calculate_practical_performance(df)
    weighted_metrics_df = calculate_regime_weighted_averages(regime_metrics_df)
    conditional_df = calculate_conditional_performance(regime_metrics_df)
    consistency_df = calculate_cross_regime_consistency(regime_metrics_df)
    
    # Calculate comprehensive comparison tables
    comprehensive_tables = calculate_comprehensive_algorithm_comparison(regime_metrics_df)
    topology_summaries = calculate_topology_regime_summary(regime_metrics_df)
    
    # Generate comprehensive summary
    summary_text = generate_regime_based_summary(
        regime_metrics_df, practical_metrics_df, weighted_metrics_df, 
        conditional_df, consistency_df
    )
    
    # Save all results
    saved_files = save_regime_based_analysis(
        regime_metrics_df, practical_metrics_df, weighted_metrics_df,
        conditional_df, consistency_df, summary_text, comprehensive_tables, topology_summaries
    )
    
    print(f"\nRegime-based analysis complete! Files saved:")
    for i, path in enumerate(saved_files, 1):
        print(f"{i}. {path}")
    
    # Print key findings about WTF performance issue
    print(f"\n=== KEY FINDINGS: WTF REGIME-DEPENDENT PERFORMANCE ===")
    
    # Show WTF performance across regimes
    wtf_regime_data = regime_metrics_df[regime_metrics_df['friendly_name'] == 'wtf']
    if len(wtf_regime_data) > 0:
        print("WTF Performance by Stubbornness Regime:")
        for regime in ['low', 'medium', 'high']:
            regime_data = wtf_regime_data[wtf_regime_data['stubbornness_regime'] == regime]
            if len(regime_data) > 0:
                mean_coop = regime_data['mean_cooperation'].mean()
                print(f"  {regime.title()} stubbornness: {mean_coop:.3f}")
    
    # Show practical vs weighted performance comparison
    wtf_weighted = weighted_metrics_df[weighted_metrics_df['friendly_name'] == 'wtf']
    wtf_practical = practical_metrics_df[practical_metrics_df['friendly_name'] == 'wtf']
    
    if len(wtf_weighted) > 0 and len(wtf_practical) > 0:
        print(f"\nWTF Performance Comparison:")
        print(f"  Regime-weighted average: {wtf_weighted['weighted_mean_cooperation'].mean():.3f}")
        print(f"  Practical performance (medium stubbornness): {wtf_practical['practical_mean_cooperation'].mean():.3f}")
        print(f"  Performance bias from extreme regimes: {wtf_weighted['weighted_mean_cooperation'].mean() - wtf_practical['practical_mean_cooperation'].mean():+.3f}")
    
    return regime_metrics_df, practical_metrics_df, weighted_metrics_df, conditional_df, consistency_df, comprehensive_tables, topology_summaries

if __name__ == "__main__":
    main()