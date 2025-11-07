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
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Total simulations: {len(df)}")
    
    # Get unique stubbornness values
    stubbornness_values = sorted(df['stubbornness'].unique())
    print(f"Stubbornness range: {min(stubbornness_values):.3f} to {max(stubbornness_values):.3f}")
    print(f"Unique values: {len(stubbornness_values)}")
    print(f"Values: {[f'{v:.3f}' for v in stubbornness_values]}")
    
    # Calculate performance metrics for each stubbornness level
    # Using averaged metrics across all parameter combinations for more robust clustering
    stub_performance = []
    for stub in stubbornness_values:
        stub_data = df[df['stubbornness'] == stub]
        
        # Overall metrics (averaged across all polarisingNode_f values and simulations)
        mean_cooperation = stub_data['state'].mean()
        mean_polarization = stub_data['state_std'].mean()
        coop_ratio = (stub_data['state'] > 0).mean()
        
        # Cooperative states only
        coop_only = stub_data[stub_data['state'] > 0]
        mean_coop_given_coop = coop_only['state'].mean() if len(coop_only) > 0 else 0
        
        # Additional stability metrics
        std_cooperation = stub_data['state'].std()
        std_polarization = stub_data['state_std'].std()
        
        stub_performance.append({
            'stubbornness': stub,
            'mean_cooperation': mean_cooperation,
            'std_cooperation': std_cooperation,
            'mean_cooperation_coop_only': mean_coop_given_coop,
            'cooperative_ratio': coop_ratio,
            'mean_polarization': mean_polarization,
            'std_polarization': std_polarization,
            'n_total': len(stub_data),
            'n_cooperative': len(coop_only)
        })
    
    perf_df = pd.DataFrame(stub_performance)
    
    print(f"\n=== PERFORMANCE BY STUBBORNNESS LEVEL ===")
    for _, row in perf_df.iterrows():
        print(f"Stubborness {row['stubbornness']:.3f}: Coop={row['mean_cooperation']:.3f}, Coop ratio={row['cooperative_ratio']:.3f}, Polarization={row['mean_polarization']:.3f}")
    
    # Prepare data for K-means clustering
    # Use multiple performance metrics to find natural groupings
    features = ['stubbornness', 'mean_cooperation', 'cooperative_ratio', 'mean_polarization', 'std_cooperation']
    clustering_data = perf_df[features].values
    
    # Standardize features for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    print(f"\n=== K-MEANS CLUSTERING ANALYSIS ===")
    
    # Test different numbers of clusters to find optimal
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = {}
    cluster_results = {}
    
    for n_clusters in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Calculate silhouette score for cluster quality
        if n_clusters > 1:
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores[n_clusters] = silhouette_avg
        
        cluster_results[n_clusters] = cluster_labels
        
        print(f"\n{n_clusters} clusters (Silhouette Score: {silhouette_scores.get(n_clusters, 'N/A'):.3f}):")
        
        # Add cluster labels to dataframe
        temp_df = perf_df.copy()
        temp_df['cluster'] = cluster_labels
        
        # Show cluster boundaries sorted by stubbornness
        cluster_summaries = []
        for cluster_id in range(n_clusters):
            cluster_data = temp_df[temp_df['cluster'] == cluster_id]
            stub_min = cluster_data['stubbornness'].min()
            stub_max = cluster_data['stubbornness'].max()
            stub_mean = cluster_data['stubbornness'].mean()
            mean_coop = cluster_data['mean_cooperation'].mean()
            mean_ratio = cluster_data['cooperative_ratio'].mean()
            
            cluster_summaries.append({
                'id': cluster_id, 'stub_min': stub_min, 'stub_max': stub_max,
                'stub_mean': stub_mean, 'mean_coop': mean_coop, 'mean_ratio': mean_ratio
            })
        
        # Sort by average stubbornness for logical ordering
        cluster_summaries.sort(key=lambda x: x['stub_mean'])
        
        for i, summary in enumerate(cluster_summaries):
            print(f"  Regime {i+1} (Cluster {summary['id']}): Stubbornness {summary['stub_min']:.3f}-{summary['stub_max']:.3f}, "
                  f"Avg cooperation {summary['mean_coop']:.3f}, Avg coop ratio {summary['mean_ratio']:.3f}")
    
    # Focus on 4 clusters (based on visualization showing 4 natural splits)
    print(f"\n=== DETAILED 4-CLUSTER ANALYSIS (RECOMMENDED) ===")
    kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels_4 = kmeans_4.fit_predict(scaled_data)
    
    perf_df['cluster'] = cluster_labels_4
    
    # Calculate 4-cluster statistics
    cluster_stats = []
    regime_names = ['Very Low', 'Low', 'Medium', 'High']
    
    # Sort clusters by stubbornness level first
    cluster_summary_4 = []
    for cluster_id in range(4):
        cluster_data = perf_df[perf_df['cluster'] == cluster_id]
        cluster_summary_4.append({
            'original_id': cluster_id,
            'stubbornness_mean': cluster_data['stubbornness'].mean(),
            'data': cluster_data
        })
    
    cluster_summary_4.sort(key=lambda x: x['stubbornness_mean'])
    
    for i, cluster_info in enumerate(cluster_summary_4):
        cluster_data = cluster_info['data']
        cluster_stats.append({
            'regime': regime_names[i],
            'original_cluster_id': cluster_info['original_id'],
            'stubbornness_min': cluster_data['stubbornness'].min(),
            'stubbornness_max': cluster_data['stubbornness'].max(),
            'stubbornness_mean': cluster_data['stubbornness'].mean(),
            'mean_cooperation': cluster_data['mean_cooperation'].mean(),
            'std_cooperation': cluster_data['std_cooperation'].mean(),
            'mean_coop_ratio': cluster_data['cooperative_ratio'].mean(),
            'mean_polarization': cluster_data['mean_polarization'].mean(),
            'std_polarization': cluster_data['std_polarization'].mean(),
            'n_levels': len(cluster_data)
        })
    
    print(f"\nNatural breaks from 4-cluster K-means analysis:")
    for i, stats in enumerate(cluster_stats):
        print(f"\n{stats['regime']} Regime (Original Cluster {stats['original_cluster_id']}):")
        print(f"  Stubbornness range: {stats['stubbornness_min']:.3f} - {stats['stubbornness_max']:.3f}")
        print(f"  Mean stubbornness: {stats['stubbornness_mean']:.3f}")
        print(f"  Mean cooperation: {stats['mean_cooperation']:.3f} (±{stats['std_cooperation']:.3f})")
        print(f"  Mean cooperative ratio: {stats['mean_coop_ratio']:.3f}")
        print(f"  Mean polarization: {stats['mean_polarization']:.3f} (±{stats['std_polarization']:.3f})")
        print(f"  Number of parameter levels: {stats['n_levels']}")
    
    # Compare with current 3-regime boundaries
    current_regimes = [(0.0, 0.4), (0.4, 0.7), (0.7, 1.0)]
    print(f"\n=== COMPARISON WITH CURRENT 3-REGIME BOUNDARIES ===")
    print(f"Current 3-regime boundaries:")
    for i, (low, high) in enumerate(current_regimes):
        print(f"  {['Low', 'Medium', 'High'][i]} regime: {low} - {high}")
    
    print(f"\nK-means suggested 4-regime boundaries:")
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