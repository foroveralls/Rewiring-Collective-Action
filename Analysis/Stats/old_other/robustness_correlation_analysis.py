#!/usr/bin/env python3
"""
Correlation analysis for robustness metrics to identify meaningful subgroups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def analyze_correlations_by_subgroup(df, x_col='cooperative_volume_percent', y_col='stubbornness_sensitivity'):
    """Analyze correlations within different subgroups"""
    
    results = []
    
    # Filter valid data
    valid_mask = (
        (df[x_col] > 0) & (df[y_col] >= 0) &
        np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    )
    df_clean = df[valid_mask].copy()
    
    print(f"\nCorrelation Analysis: {y_col} vs {x_col}")
    print("="*60)
    
    # Overall correlation
    overall_r, overall_p = stats.pearsonr(df_clean[x_col], df_clean[y_col])
    print(f"Overall: r={overall_r:.3f}, p={overall_p:.3f}, n={len(df_clean)}")
    
    # By topology
    print("\nBy Topology:")
    for topology in df_clean['topology'].unique():
        subset = df_clean[df_clean['topology'] == topology]
        if len(subset) >= 3:
            r, p = stats.pearsonr(subset[x_col], subset[y_col])
            print(f"  {topology}: r={r:.3f}, p={p:.3f}, n={len(subset)}")
            results.append({
                'group_type': 'topology', 'group': topology, 
                'r': r, 'p': p, 'n': len(subset)
            })
    
    # By algorithm
    print("\nBy Algorithm:")
    for algo in df_clean['friendly_name'].unique():
        subset = df_clean[df_clean['friendly_name'] == algo]
        if len(subset) >= 3:
            r, p = stats.pearsonr(subset[x_col], subset[y_col])
            print(f"  {algo}: r={r:.3f}, p={p:.3f}, n={len(subset)}")
            results.append({
                'group_type': 'algorithm', 'group': algo,
                'r': r, 'p': p, 'n': len(subset)
            })
    
    # By topology-algorithm combinations
    print("\nBy Topology-Algorithm combinations:")
    for topology in df_clean['topology'].unique():
        for algo in df_clean['friendly_name'].unique():
            subset = df_clean[(df_clean['topology'] == topology) & 
                            (df_clean['friendly_name'] == algo)]
            if len(subset) >= 3:
                r, p = stats.pearsonr(subset[x_col], subset[y_col])
                print(f"  {topology}-{algo}: r={r:.3f}, p={p:.3f}, n={len(subset)}")
                results.append({
                    'group_type': 'combo', 'group': f"{topology}-{algo}",
                    'r': r, 'p': p, 'n': len(subset)
                })
    
    return pd.DataFrame(results)

def perform_pca_analysis(df):
    """Perform PCA to identify main sources of variation"""
    
    # Select numerical columns for PCA
    numerical_cols = [
        'cooperative_volume_percent', 'stubbornness_sensitivity', 'backfirer_sensitivity',
        'mean_cooperation', 'cooperative_ratio', 'mean_polarization'
    ]
    
    # Filter to available columns and remove NaN
    available_cols = [col for col in numerical_cols if col in df.columns]
    df_pca = df[available_cols].dropna()
    
    if len(df_pca) < 10:
        print("Not enough data for PCA analysis")
        return None, None, None
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Print explained variance
    print("\nPCA Analysis:")
    print("="*40)
    for i, var_ratio in enumerate(pca.explained_variance_ratio_[:4]):
        print(f"PC{i+1}: {var_ratio:.3f} ({var_ratio*100:.1f}%)")
    
    print(f"\nCumulative variance explained by first 3 PCs: {pca.explained_variance_ratio_[:3].sum():.3f}")
    
    # Print component loadings
    print("\nComponent Loadings (first 3 PCs):")
    loadings_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=available_cols
    )
    print(loadings_df.round(3))
    
    return pca, X_pca, loadings_df

def main():
    """Main analysis function"""
    
    # Load data
    data_path = "../../Output/Stats/stubborness_backfirer/heatmap_metrics_detailed_20250916.csv"
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} data points")
    print(f"Topologies: {df['topology'].unique()}")
    print(f"Algorithms: {df['friendly_name'].unique()}")
    
    # Analyze correlations for both sensitivity metrics
    for metric in ['stubbornness_sensitivity', 'backfirer_sensitivity']:
        if metric in df.columns:
            results = analyze_correlations_by_subgroup(df, y_col=metric)
            
            # Find strongest correlations
            strong_corr = results[abs(results['r']) > 0.5]
            if len(strong_corr) > 0:
                print(f"\nStrongest correlations for {metric}:")
                for _, row in strong_corr.iterrows():
                    print(f"  {row['group']}: r={row['r']:.3f}")
    
    # PCA analysis
    pca, X_pca, loadings = perform_pca_analysis(df)
    
    # Identify which groupings to focus on
    print("\nRecommendations:")
    print("="*40)
    
    if pca is not None:
        # Check if basin stability and sensitivity load on same PC
        basin_pc1 = abs(loadings.loc['cooperative_volume_percent', 'PC1'])
        stub_pc1 = abs(loadings.loc['stubbornness_sensitivity', 'PC1'])
        
        if basin_pc1 > 0.5 and stub_pc1 > 0.5:
            print("- Basin stability and sensitivity load strongly on PC1 - may have underlying relationship")
        else:
            print("- Basin stability and sensitivity load on different PCs - relationship may be topology/algorithm dependent")
    
    # Check for topology-specific patterns
    topo_results = results[results['group_type'] == 'topology']
    if len(topo_results) > 0:
        best_topo = topo_results.loc[topo_results['r'].abs().idxmax()]
        print(f"- Strongest topology-specific correlation: {best_topo['group']} (r={best_topo['r']:.3f})")
    
    # Check for algorithm-specific patterns  
    algo_results = results[results['group_type'] == 'algorithm']
    if len(algo_results) > 0:
        best_algo = algo_results.loc[algo_results['r'].abs().idxmax()]
        print(f"- Strongest algorithm-specific correlation: {best_algo['group']} (r={best_algo['r']:.3f})")

if __name__ == "__main__":
    main()