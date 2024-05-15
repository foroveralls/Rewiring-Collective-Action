# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:09:27 2024

@author: Jordan
"""
import networkx as nx
from netin import viz
from netin import DPAH, PATCH

#%% testing and building networks

G = DPAH(n=50, f_m=0.5, d=0.1, h_MM=0.5, h_mm=0.5, plo_M=2.0, plo_m=2.0, seed=42)
G.generate()
viz.plot_graph(G, edge_arrows = True, edge_width = 1, arrow_size =10, cell_size = 3, node_size = 50)


#minority = 1, majority 2
G.nodes[1]["m"]

#%%

# Example of loading a graph from an edge list file
#G = nx.read_edgelist('path_to_your_file.edgelist', create_using=nx.DiGraph())  # Use nx.Graph() for undirected graphs

G = nx.barabasi_albert_graph(100, 4)
# Manually assigning minority or majority status
#for node in G.nodes():
 #   G.nodes[node]['status'] = 'minority' if node_criteria(node) else 'majority'

# Initialize DPAH model
#dpah_model = DPAH(G, minority_fraction=0.3)  # adjust minority_fraction as needed

# Run the model for a number of steps
#dpah_model.run(steps=50)  # adjust steps based on your requirements

# For PATCH, the approach is similar
patch_model = PATCH(
patch_model.run(steps=50)

# Analyzing the graph
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Visualizing the graph
nx.draw(G, with_labels=True)
plt.show()
