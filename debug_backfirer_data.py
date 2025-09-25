#!/usr/bin/env python3
"""
Debug script to analyze backfirer regime data distribution and identify missing data points
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def analyze_backfirer_data():
    """Analyze the distribution of backfirer fraction values"""
    base_dir = "Output/Stats/stubborness_backfirer"
    
    # Load data from all regimes
    all_data = []
    algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
    
    for regime in ['low', 'medium', 'high']:
        file_path = f"{base_dir}/comprehensive_algorithm_comparison_{regime}_20250924.csv"
        df = pd.read_csv(file_path)
        
        print(f"\n{regime.upper()} REGIME:")
        for alg in algorithms:
            if alg in df.columns:
                backfirer_val = df[df['Metric'] == 'Mean Backfirer Fraction'][alg].values[0]
                coop_val = df[df['Metric'] == 'Mean Cooperation'][alg].values[0]
                print(f"  {alg}: backfirer={backfirer_val:.4f}, cooperation={coop_val:.3f}")
                all_data.append({
                    'algorithm': alg,
                    'regime': regime,
                    'backfirer_fraction': backfirer_val,
                    'cooperation': coop_val
                })
    
    # Convert to DataFrame for analysis
    df_all = pd.DataFrame(all_data)
    
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total data points: {len(df_all)}")
    print(f"Backfirer fraction range: {df_all['backfirer_fraction'].min():.4f} to {df_all['backfirer_fraction'].max():.4f}")
    print(f"Mean: {df_all['backfirer_fraction'].mean():.4f}")
    print(f"Std: {df_all['backfirer_fraction'].std():.4f}")
    
    # Apply current thresholds
    low_threshold = 0.043
    high_threshold = 0.097
    
    print(f"\n=== CURRENT THRESHOLDS ===")
    print(f"Low threshold: {low_threshold}")
    print(f"High threshold: {high_threshold}")
    
    # Classify data points
    low_count = sum(df_all['backfirer_fraction'] <= low_threshold)
    medium_count = sum((df_all['backfirer_fraction'] > low_threshold) & (df_all['backfirer_fraction'] <= high_threshold))
    high_count = sum(df_all['backfirer_fraction'] > high_threshold)
    
    print(f"\nData point distribution:")
    print(f"  Low regime (<={low_threshold}): {low_count} points")
    print(f"  Medium regime ({low_threshold}-{high_threshold}): {medium_count} points") 
    print(f"  High regime (>{high_threshold}): {high_count} points")
    
    # Show which algorithms/regimes fall into each category
    print(f"\n=== DETAILED CLASSIFICATION ===")
    for backfirer_regime in ['Low', 'Medium', 'High']:
        if backfirer_regime == 'Low':
            subset = df_all[df_all['backfirer_fraction'] <= low_threshold]
        elif backfirer_regime == 'Medium':
            subset = df_all[(df_all['backfirer_fraction'] > low_threshold) & (df_all['backfirer_fraction'] <= high_threshold)]
        else:
            subset = df_all[df_all['backfirer_fraction'] > high_threshold]
        
        print(f"\n{backfirer_regime} backfirer regime:")
        for _, row in subset.iterrows():
            print(f"  {row['algorithm']} ({row['regime']} stubbornness): {row['backfirer_fraction']:.4f}")
    
    # Check for algorithms with missing data in any backfirer regime
    print(f"\n=== ALGORITHM COVERAGE ANALYSIS ===")
    for alg in algorithms:
        alg_data = df_all[df_all['algorithm'] == alg]
        
        # Check which backfirer regimes this algorithm appears in
        low_present = any(alg_data['backfirer_fraction'] <= low_threshold)
        medium_present = any((alg_data['backfirer_fraction'] > low_threshold) & (alg_data['backfirer_fraction'] <= high_threshold))
        high_present = any(alg_data['backfirer_fraction'] > high_threshold)
        
        missing_regimes = []
        if not low_present: missing_regimes.append('Low')
        if not medium_present: missing_regimes.append('Medium')
        if not high_present: missing_regimes.append('High')
        
        if missing_regimes:
            print(f"  {alg}: MISSING from {', '.join(missing_regimes)} backfirer regime(s)")
        else:
            print(f"  {alg}: Present in all backfirer regimes")

if __name__ == "__main__":
    analyze_backfirer_data()