#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:20:15 2025

@author: jpoveralls
"""

#!/usr/bin/env python3
"""
Network analysis for empirical networks
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_network(G, name):
    """Calculate standard network metrics"""
    n, m = G.number_of_nodes(), G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    
    # Basic metrics
    metrics = {
        'Network': name,
        'N': n,
        'M': m,
        'Density': nx.density(G),
        'Avg_Degree': np.mean(degrees),
        'Clustering': nx.average_clustering(G)
    }
    
    # Connectivity
    if nx.is_directed(G):
        comps = list(nx.strongly_connected_components(G))
        giant = G.subgraph(max(comps, key=len))
    else:
        comps = list(nx.connected_components(G))
        giant = G.subgraph(max(comps, key=len))
    
    # Path metrics
    try:
        metrics['Avg_Path_Length'] = nx.average_shortest_path_length(giant)
    except:
        metrics['Avg_Path_Length'] = np.nan
    
    # Advanced metrics
    try:
        metrics['Assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        metrics['Assortativity'] = np.nan
    
    try:
        partition = nx.community.greedy_modularity_communities(G)
        metrics['Modularity'] = nx.community.modularity(G, partition)
        metrics['Communities'] = len(partition)
    except:
        metrics['Modularity'] = np.nan
        metrics['Communities'] = np.nan
    
    # Small-world coefficient
    try:
        if nx.is_directed(G):
            G_rand = nx.erdos_renyi_graph(n, m/(n*(n-1)), directed=True)
        else:
            G_rand = nx.erdos_renyi_graph(n, 2*m/(n*(n-1)))
        
        C = nx.average_clustering(G)
        C_rand = nx.average_clustering(G_rand)
        L = metrics['Avg_Path_Length']
        L_rand = nx.average_shortest_path_length(G_rand)
        
        metrics['Small_World_Sigma'] = (C/C_rand) / (L/L_rand)
    except:
        metrics['Small_World_Sigma'] = np.nan
    
    # Power law fit
    deg_counts = np.bincount(degrees)[1:]
    x = np.arange(1, len(deg_counts) + 1)
    y = deg_counts[deg_counts > 0]
    x = x[deg_counts > 0]
    if len(x) > 3:
        slope, _, r, _, _ = stats.linregress(np.log(x), np.log(y))
        metrics['Power_Law_Alpha'] = -slope
    else:
        metrics['Power_Law_Alpha'] = np.nan
    
    return metrics

def main():
    """Load networks and analyze"""
    results = []
    
    for f in Path("../Pre_processing/networks_processed").glob("*.gpickle"):
        name = f.stem.replace('_graph', '').replace('_N_', '_')
        G = nx.read_gpickle(f)
        print(f"Analyzing {name}: N={G.number_of_nodes()}, M={G.number_of_edges()}")
        metrics = analyze_network(G, name)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df.to_csv("network_properties.csv", index=False)
    
    print("\nResults:")
    cols = ['Network', 'N', 'M', 'Avg_Degree', 'Clustering', 'Modularity', 
            'Assortativity', 'Small_World_Sigma', 'Power_Law_Alpha']
    print(df[cols].round(3).to_string(index=False))
    
    return df

if __name__ == "__main__":
    df = main()