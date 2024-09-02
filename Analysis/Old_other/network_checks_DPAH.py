# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:09:27 2024

@author: Jordan
"""
import networkx as nx
from netin import viz
from netin import DPAH, PATCH, DH
from netin.generators.h import Homophily
from models_checks_updated import plot_network
#import network_stats

#%% testing and building networks

G = DPAH(n=500, f_m=0.5, d=0.02, h_MM=0.5, h_mm=0.5, plo_M=2.0, plo_m=2.0, seed=42)
G.generate()
viz.plot_graph(G, edge_arrows = True, edge_width = 1, arrow_size =10, cell_size = 3, node_size = 50)


layout = nx.spring_layout(G, k=0.3, iterations=50)
nx.draw(G, pos=layout, arrows=nx.is_directed(G), with_labels=False, edge_color='gray', edgecolors = "black", node_size=190, 
        font_size=10, alpha=0.9)

nx.draw(G)

print(Homophily.infer_homophily_values(G))
#print(network_stats.infer_homophily_values(G, G.calculate_fraction_of_minority()))

#%%


# #minority = 1, majority 2
# G.nodes[1]["m"]

G = DH(n=200, f_m=0.5, d=0.1, h_MM=0.5, h_mm=0.5, plo_M=2.0, plo_m=2.0, seed=42)
G.generate()
print(G.infer_homophily_values())
#print(network_stats.infer_homophily_values(G, G.calculate_fraction_of_minority()))

#viz.plot_graph(G, edge_arrows = True, edge_width = 1, arrow_size =10, cell_size = 3, node_size = 50)


G = PATCH(n=300, f_m=0.5, h_MM=0.5, h_mm=0.5, k =4, tc= 0.5, seed=42)
G.generate()
viz.plot_graph(G, edge_arrows = True, edge_width = 1, arrow_size =10, cell_size = 3, node_size = 50)
print(Homophily.infer_homophily_values(G))
#print(network_stats.infer_homophily_values(G, G.calculate_fraction_of_minority()))
#print(G.infer_homophily_values())

#%%
