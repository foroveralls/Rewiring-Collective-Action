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
import community as community_louvain
import igraph as ig
import leidenalg as la

#dir_path = 'Data'

#%% Define functions
def load_and_analyze_edges(folder_path, network_type):
    """
    Generates a network graph from a given edge file.
    Args:
    - file_path: Path to the edge file.
    - network_type: Type of network ('facebook' for undirected, 'twitter' for directed mutual follows).
    Returns:
    - G: A NetworkX graph object.
    """
    # Placeholder for our statistics
    stats_list = []
    graph_list = []
    
    folder_path = Path(folder_path)
    
    # Iterating over .edges files in the given folder
    for edge_file in glob(os.path.join(folder_path, '*.edge*')):
        
        
        df_edges = pd.read_csv(edge_file, delim_whitespace=True, header=None, names=['node1', 'node2'])
    
        # For Facebook or similar: create an undirected graph from edges
        G = nx.from_pandas_edgelist(df_edges, 'node1', 'node2')
    
        if 'twitter' in network_type:
           # For Twitter: create a directed graph and then find mutual follows to make undirected
           G = nx.from_pandas_edgelist(df_edges, 'node1', 'node2', create_using=nx.DiGraph)
           # mutual_follows = [(u, v) for u, v in G_directed.edges() if G_directed.has_edge(v, u)]
           # G = nx.Graph() # Create an empty undirected graph
           # G.add_edges_from(mutual_follows) # Add only mutual follows as undirected edges
            
        # Calculate statistics
        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        avg_degree = sum(dict(G.degree()).values()) / float(num_nodes)
        density = nx.density(G)
        directed = nx.is_directed(G)
        
        # Append statistics to list
        stats_list.append({
            'file_name': Path(edge_file).name,
            'num_edges': num_edges,
            'num_nodes': num_nodes,
            'avg_degree': avg_degree,
            'density': density,
            'directed': directed
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

def create_connected_subgraph(folder_path, N):
    G_all = nx.Graph()

    # Parse through all .edges files in the folder
    for edge_file_path in Path(folder_path).glob('*.edges'):
        df_edges = pd.read_csv(edge_file_path, delim_whitespace=True, header=None, names=['node1', 'node2'])
        
        # Create a directed graph to find mutual follows
        G_directed = nx.from_pandas_edgelist(df_edges, 'node1', 'node2', create_using=nx.DiGraph())
        
        # Add mutual follows as undirected edges to the main graph
        for u, v in G_directed.edges():
            if G_directed.has_edge(v, u):
                G_all.add_edge(u, v)
    
    
    
    return find_largest_component(G_all, N)

def find_largest_component(G_all, N):        
    """
    Finds largest connected component in network and returns it as a subgraph
    """     
    # Find the largest connected component
    largest_cc = max(nx.connected_components(G_all), key=len)
    subgraph = G_all.subgraph(largest_cc).copy()
    
    
    # Check if the resulting subgraph is connected and has N nodes
    # assert nx.is_connected(subgraph), "Resulting subgraph is not connected."
    # assert len(subgraph) == N, f"Resulting subgraph does not have {N} nodes. {len(subgraph)}"
    
    return subgraph


def load_network_from_edges(file_path):
    """
    Loads a network graph from a given edge file.
    """
    df_edges = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['node1', 'node2'])
    G = nx.from_pandas_edgelist(df_edges, 'node1', 'node2')
    return G

def community_based_reduction(G, N):
    """
    Reduces the size of the network G to approximately N nodes by sampling from its communities, 
    ensuring the resulting subgraph remains connected. If disconnected components result from 
    sampling, the smaller components are removed until the graph is connected.
    """
    
    def community_detection_with_leidenalg(nx_graph):
        # Convert networkx graph to igraph
        ig_graph = ig.Graph.from_networkx(nx_graph)
        
        # Perform community detection using leidenalg on the igraph graph
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
        
        # Map the community detection results back to the networkx graph
        # Creating a dictionary where keys are nodes in the original networkx graph,
        # and values are their community labels
        nx_partition = {}
        for node in nx_graph.nodes():
            #ig_index = ig_graph.vs.find(name=str(node)).index  # Find the igraph index of the node
            nx_partition[node] = partition.membership[node]
        
        return nx_partition
    # Detect communities with the Louvain method
    partition = community_detection_with_leidenalg(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    
    # Initialize variables for the reduction process
    H = None
    adjustment_factor = 0.1  # Initial adjustment to samples per community if needed
    
    while True:
        sampled_nodes = []

        # Adjust the number of nodes to sample from each community, based on previous iterations
        total_nodes = sum(len(comm) for comm in communities.values())
        samples_per_comm = {comm_id: max(1, int(len(nodes) / total_nodes * (N + N * adjustment_factor)))
                            for comm_id, nodes in communities.items()}
        
        # Sample nodes from each community
        for comm_id, nodes in communities.items():
            sampled_nodes.extend(nodes[:samples_per_comm[comm_id]])
        
        # Create a subgraph based on the sampled nodes
        H = G.subgraph(sampled_nodes)
        
        # Check connectivity and adjust if necessary
        if nx.is_connected(H):
            if len(H) >= N * 0.85 and len(H) <= N * 1.10:  # Check if size is within 15% of N
                break  # The graph meets our conditions
            else:
                adjustment_factor -= 0.05  # Adjust sampling for size correction
        else:
            # Remove smaller disconnected components
            components = list(nx.connected_components(H))
            largest_component = max(components, key=len)
            H = G.subgraph(largest_component).copy()  # Keep only the largest component
            if len(H) <= N * 1.15:  # If after removal we're close to desired size, we stop
                break
            adjustment_factor -= 0.05  # Adjust sampling for next iteration

    # Additional step: If needed, further nodes can be pruned to closely match N
    
    return H


def community_based_reduction_directed(G, N):
    """
    Reduces the size of the directed network G to approximately N nodes by sampling from its communities, 
    ensuring the resulting subgraph remains strongly connected. If disconnected components result from 
    sampling, the smaller components are removed until the graph is strongly connected.
    """

    def community_detection_with_leidenalg(nx_graph):
        # Convert networkx graph to igraph
        ig_graph = ig.Graph.from_networkx(nx_graph)
        
        # Perform community detection using leidenalg on the igraph graph
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
        
        # Map the community detection results back to the networkx graph
        nx_partition = {}
        for node in nx_graph.nodes():
            nx_partition[node] = partition.membership[node]
        
        return nx_partition

    # Detect communities with the leidenalg method
    partition = community_detection_with_leidenalg(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    
    H = None
    adjustment_factor = 0.1  # Initial adjustment to samples per community if needed
    
    while True:
        sampled_nodes = []

        # Adjust the number of nodes to sample from each community, based on previous iterations
        total_nodes = sum(len(comm) for comm in communities.values())
        samples_per_comm = {comm_id: max(1, int(len(nodes) / total_nodes * (N + N * adjustment_factor)))
                            for comm_id, nodes in communities.items()}
        
        # Sample nodes from each community
        for comm_id, nodes in communities.items():
            sampled_nodes.extend(nodes[:samples_per_comm[comm_id]])
        
        # Create a subgraph based on the sampled nodes
        H = G.subgraph(sampled_nodes).copy()
        
        # Check for strong connectivity and adjust if necessary
        if nx.is_strongly_connected(H):
            if len(H) >= N * 0.85 and len(H) <= N * 1.10:  # Check if size is within 15% of N
                break  # The graph meets our conditions
            else:
                adjustment_factor -= 0.05  # Adjust sampling for size correction
        else:
            # Remove smaller disconnected components
            components = list(nx.strongly_connected_components(H))
            largest_component = max(components, key=len)
            H = G.subgraph(largest_component).copy()  # Keep only the largest component
            if len(H) <= N * 1.15:  # If after removal we're close to desired size, we stop
                break
            adjustment_factor -= 0.05  # Adjust sampling for next iteration
    
    return H


def remap(G1):
    
    node_mapping = {old_name: new_name for new_name, old_name in enumerate(G1.nodes())}
    # Apply the mapping to the graph
    G2 = nx.relabel_nodes(G1, node_mapping)
    return G2

def reduce_graph(G, N):
    
    subgraph = find_largest_component(G, N)
    
    # If the largest connected component is bigger than N, select a subgraph of exactly N nodes
    if len(subgraph) > N*1.15:
        subgraph = community_based_reduction (subgraph, N)
   
    return subgraph

#%% run analysis

stats_list, graph_list = [], []

folder_path = "Data"
for i in os.listdir(folder_path):
    j = os.path.join(folder_path, i)
    stat, graph = load_and_analyze_edges(j, network_type= i)
    stats_list.append(stat); graph_list.append(graph)

#%%

#components = list(nx.connected_components(graph[0]))
N = 100
#G_reduced_1 = find_largest_component(graph[0], 100)

#G_reduced = community_based_reduction(graph[], N)
#G_reduced = community_based_reduction_directed(graph[0], N)


#nx.draw(G_reduced, node_size = 10)
#G = G_reduced

#save graph
#nx.write_gpickle(G, f'networks_processed/_graph_N_{G.number_of_nodes()}.gpickle')
#T = nx.read_gpickle("networks_processed/twitter_graph_N_891.gpickle")
#%%

# edge_file = "Data/facebook_test/fb_edgelist.txt"
# df_edges = pd.read_csv(edge_file, delim_whitespace=True, header=None, names=['node1', 'node2'])



# # For Facebook or similar: create an undirected graph from edges
# G1 = nx.from_pandas_edgelist(df_edges, 'node1', 'node2')
# G1 = remap(G1)
# nx.write_gpickle(G1, f'networks_processed/twitter_graph_N_{G1.number_of_nodes()}.gpickle')





