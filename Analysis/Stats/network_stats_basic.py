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
import pickle
import networkx as nx
import numpy as np
from netin import DPAH
from pathlib import Path
from scipy import stats
import igraph as ig
import leidenalg as la
import warnings
warnings.filterwarnings('ignore')


def leiden_partition(G):
    """Leiden communities as a list of node sets, for nx.community.modularity.

    Built from the edge list rather than ig.Graph.from_networkx() so igraph does not
    copy node attributes, and indexed by igraph vertex id rather than by networkx
    node label (the two coincide only for 0..N-1 labels in insertion order).
    Direction is preserved, so directed graphs get the directed modularity.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    ig_g = ig.Graph(n=len(nodes), edges=edges, directed=G.is_directed())
    part = la.find_partition(ig_g, la.ModularityVertexPartition, seed=42)
    return [{nodes[i] for i in comm} for comm in part]


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
    
    # Leiden, seed=42, unweighted: the same call the model makes
    # (models_checks.py:1430) and the only method the manuscript names (L246, L394).
    # Was greedy_modularity_communities until 2026-07-30, which is why the Q values
    # in this table sat below the ones in SI Fig. S2 (Louvain): on FB at t=0,
    # 0.5205 greedy vs 0.5759 Louvain vs 0.5773 Leiden.
    try:
        partition = leiden_partition(G)
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
    
    #empirical
    for f in Path("../../Pre_processing/networks_processed").glob("*.gpickle"):
        name = f.stem.replace('_graph', '').replace('_N_', '_')
        #G = nx.read_gpickle(f)
        with open(f, 'rb') as f:
            G = pickle.load(f)
        print(f"Analyzing {name}: N={G.number_of_nodes()}, M={G.number_of_edges()}")
        metrics = analyze_network(G, name)
        results.append(metrics)
    
    #synthetic, has a modulartity of approx 0.20
    G_dpah = DPAH(n=800, f_m=0.5, d=0.02, h_MM=0.5, h_mm=0.5, plo_M=2.0, plo_m=2.0, seed = 42)
    
    G_dpah.generate()
    # DiGraph, not Graph: DPAH is directed and measuring it on an undirected
    # projection made this row incomparable with the directed Twitter row above.
    # (Plain nx class rather than the netin object because community detection and
    # several nx algorithms reconstruct the graph class, which netin cannot do.)
    G = nx.DiGraph(G_dpah)
    results.append(analyze_network(G, "DPAH_graph"))
    
    df = pd.DataFrame(results)
    df.to_csv("network_properties.csv", index=False)
    
    print("\nResults:")
    cols = ['Network', 'N', 'M', 'Avg_Degree', 'Clustering', 'Modularity', 
            'Assortativity', 'Small_World_Sigma', 'Power_Law_Alpha']
    print(df[cols].round(3).to_string(index=False))
    
    return df

if __name__ == "__main__":
    df = main()