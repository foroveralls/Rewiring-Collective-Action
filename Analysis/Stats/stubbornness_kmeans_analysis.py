#!/usr/bin/env python3
"""
K-means clustering analysis for stubbornness regime determination
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_stubbornness_clustering():
    """Perform K-means clustering on stubbornness data to find natural breaks"""
    
    # Load raw data
    file_path = "../../Output/heatmap_sweep_phased_sweep_20250904_1545_stubbornness_polarisingNode_f_ihp.csv"
    df = pd.read_csv(file_path)
    
    # Get unique stubbornness values
    stubbornness_values = sorted(df['stubbornness'].unique())
    print(f"Stubbornness range: {min(stubbornness_values):.3f} to {max(stubbornness_values):.3f}")
    print(f"Unique values: {len(stubbornness_values)}")
    print(f"Values: {[f'{v:.3f}' for v in stubbornness_values]}")
    
    # Calculate performance metrics for each stubbornness level
    stub_performance = []
    for stub in stubbornness_values:
        stub_data = df[df['stubbornness'] == stub]
        
        # Overall metrics
        mean_cooperation = stub_data['state'].mean()
        mean_polarization = stub_data['state_std'].mean()
        coop_ratio = (stub_data['state'] > 0).mean()
        
        # Cooperative states only
        coop_only = stub_data[stub_data['state'] > 0]
        mean_coop_given_coop = coop_only['state'].mean() if len(coop_only) > 0 else 0
        
        stub_performance.append({
            'stubbornness': stub,
            'mean_cooperation': mean_cooperation,
            'mean_cooperation_coop_only': mean_coop_given_coop,
            'cooperative_ratio': coop_ratio,
            'mean_polarization': mean_polarization,
            'n_total': len(stub_data),
            'n_cooperative': len(coop_only)
        })
    
    perf_df = pd.DataFrame(stub_performance)
    
    print(f"\n=== PERFORMANCE BY STUBBORNNESS LEVEL ===")
    for _, row in perf_df.iterrows():
        print(f"Stubborness {row['stubbornness']:.3f}: Coop={row['mean_cooperation']:.3f}, Coop ratio={row['cooperative_ratio']:.3f}, Polarization={row['mean_polarization']:.3f}")
    
    # Prepare data for K-means clustering
    # Use multiple metrics to find natural groupings
    features = ['stubbornness', 'mean_cooperation', 'cooperative_ratio', 'mean_polarization']
    clustering_data = perf_df[features].values
    
    # Standardize features for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    print(f"\n=== K-MEANS CLUSTERING ANALYSIS ===")
    
    # Try different numbers of clusters
    for n_clusters in [3]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        print(f"\n{n_clusters} clusters:")
        
        # Add cluster labels to dataframe
        temp_df = perf_df.copy()
        temp_df['cluster'] = cluster_labels
        
        # Show cluster boundaries
        for cluster_id in range(n_clusters):
            cluster_data = temp_df[temp_df['cluster'] == cluster_id]
            stub_min = cluster_data['stubbornness'].min()
            stub_max = cluster_data['stubbornness'].max()
            mean_coop = cluster_data['mean_cooperation'].mean()
            mean_ratio = cluster_data['cooperative_ratio'].mean()
            
            print(f"  Cluster {cluster_id}: Stubbornness {stub_min:.3f}-{stub_max:.3f}, "
                  f"Avg cooperation {mean_coop:.3f}, Avg coop ratio {mean_ratio:.3f}")
    
    # Focus on 3 clusters (most relevant for regime analysis)
    print(f"\n=== DETAILED 3-CLUSTER ANALYSIS ===")
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels_3 = kmeans_3.fit_predict(scaled_data)
    
    perf_df['cluster'] = cluster_labels_3
    
    # Calculate cluster statistics
    cluster_stats = []
    regime_names = ['Low', 'Medium', 'High']
    
    for cluster_id in range(3):
        cluster_data = perf_df[perf_df['cluster'] == cluster_id]
        cluster_stats.append({
            'regime': regime_names[cluster_id],
            'stubbornness_min': cluster_data['stubbornness'].min(),
            'stubbornness_max': cluster_data['stubbornness'].max(),
            'stubbornness_mean': cluster_data['stubbornness'].mean(),
            'mean_cooperation': cluster_data['mean_cooperation'].mean(),
            'mean_coop_ratio': cluster_data['cooperative_ratio'].mean(),
            'mean_polarization': cluster_data['mean_polarization'].mean(),
            'n_levels': len(cluster_data)
        })
    
    # Sort by stubbornness level
    cluster_stats.sort(key=lambda x: x['stubbornness_mean'])
    
    print(f"\nNatural breaks from K-means clustering:")
    for i, stats in enumerate(cluster_stats):
        print(f"\n{stats['regime']} Regime:")
        print(f"  Stubbornness range: {stats['stubbornness_min']:.3f} - {stats['stubbornness_max']:.3f}")
        print(f"  Mean stubbornness: {stats['stubbornness_mean']:.3f}")
        print(f"  Mean cooperation: {stats['mean_cooperation']:.3f}")
        print(f"  Mean cooperative ratio: {stats['mean_coop_ratio']:.3f}")
        print(f"  Mean polarization: {stats['mean_polarization']:.3f}")
        print(f"  Number of parameter levels: {stats['n_levels']}")
    
    # Compare with current regime boundaries
    current_regimes = [(0.0, 0.4), (0.4, 0.7), (0.7, 1.0)]
    print(f"\n=== COMPARISON WITH CURRENT REGIME BOUNDARIES ===")
    print(f"Current boundaries:")
    for i, (low, high) in enumerate(current_regimes):
        print(f"  {regime_names[i]} regime: {low} - {high}")
    
    print(f"\nK-means suggested boundaries:")
    for i, stats in enumerate(cluster_stats):
        print(f"  {stats['regime']} regime: {stats['stubbornness_min']:.3f} - {stats['stubbornness_max']:.3f}")
    
    # Calculate performance differences between current and K-means regimes
    print(f"\n=== REGIME PERFORMANCE COMPARISON ===")
    
    # Current regime performance
    current_regime_stats = []
    for i, (low, high) in enumerate(current_regimes):
        regime_data = perf_df[(perf_df['stubbornness'] >= low) & (perf_df['stubbornness'] < high)]
        if i == 2:  # High regime - include upper boundary
            regime_data = perf_df[perf_df['stubbornness'] >= low]
        
        current_regime_stats.append({
            'regime': regime_names[i],
            'mean_cooperation': regime_data['mean_cooperation'].mean(),
            'mean_coop_ratio': regime_data['cooperative_ratio'].mean(),
            'n_levels': len(regime_data)
        })
    
    print(f"\nCurrent regime performance:")
    for stats in current_regime_stats:
        print(f"  {stats['regime']}: Cooperation {stats['mean_cooperation']:.3f}, "
              f"Coop ratio {stats['mean_coop_ratio']:.3f}, Levels {stats['n_levels']}")
    
    print(f"\nK-means regime performance:")
    for stats in cluster_stats:
        print(f"  {stats['regime']}: Cooperation {stats['mean_cooperation']:.3f}, "
              f"Coop ratio {stats['mean_coop_ratio']:.3f}, Levels {stats['n_levels']}")
    
    return perf_df, cluster_stats, current_regime_stats

if __name__ == "__main__":
    analyze_stubbornness_clustering()