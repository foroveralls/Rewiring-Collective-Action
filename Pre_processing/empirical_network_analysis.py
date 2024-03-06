# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:31:28 2024

@author: everall
"""

import os
import pandas as pd
import networkx as nx
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display

folder_path = 'Data/Facebook'

def load_and_analyze_edges(folder_path):
    """
    Iterates over edge files in a folder, calculates descriptive statistics for each network,
    and returns a DataFrame with these statistics.
    """
    # Placeholder for our statistics
    stats_list = []
    graph_list = []
    
    # Iterating over .edges files in the given folder
    for edge_file in glob(os.path.join(folder_path, '*.edges')):
        # Read the edge list from the file
        df_edges = pd.read_csv(edge_file, delim_whitespace=True, header=None, names=['node1', 'node2'])
        
        # Create a graph from edges
        G = nx.from_pandas_edgelist(df_edges, 'node1', 'node2')
        
        # Calculate statistics
        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        avg_degree = sum(dict(G.degree()).values()) / float(num_nodes)
        density = nx.density(G)
        
        # Append statistics to list
        stats_list.append({
            'file_name': Path(edge_file).name,
            'num_edges': num_edges,
            'num_nodes': num_nodes,
            'avg_degree': avg_degree,
            'density': density
        })
        graph_list.append(G)
        #plt.figure(figsize=(10, 8)) # Adjust the figure size as needed
        # nx.draw(G, with_labels=False, node_size=15, edge_color='gray', width = 0.3, node_color='blue', alpha=0.7)
        # plt.title(f"Network ID: {Path(edge_file)}")
        # display(plt.gcf()) # Display the current figure
        # plt.clf() # Clear the current figure to prevent overlap in the next iteration
    # Convert list of statistics to DataFrame
    stats_df = pd.DataFrame(stats_list)
    return stats_df, graph_list

def generate_network_from_file(file_path):
    """
    Generates a network graph from a given edge file.
    """
    df_edges = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['node1', 'node2'])
    G = nx.from_pandas_edgelist(df_edges, 'node1', 'node2')
    return G

# Example usage: Uncomment the following line and replace 'your_folder_path' with the actual folder path
stats_df, graph_list = load_and_analyze_edges(folder_path)
print(stats_df)


# G_example = generate_network_from_file(file_path)

# # Let's check the basic info of the generated graph to ensure it works correctly.
# G_example_info = {
#     'num_edges': G_example.number_of_edges(),
#     'num_nodes': G_example.number_of_nodes(),
#     'avg_degree': sum(dict(G_example.degree()).values()) / float(G_example.number_of_nodes()),
#     'density': nx.density(G_example)
# }

# G_example_info
