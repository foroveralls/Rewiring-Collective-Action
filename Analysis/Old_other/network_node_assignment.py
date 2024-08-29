# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:04:18 2024

@author: Jordan
"""
import networkx as nx
import numpy as np
import pandas as pd
from netin.stats.ranking import gini
from netin.viz.handlers import plot_gini_coefficient
import matplotlib.pyplot as plt


#%% Calculate metrics

def calculate_gini_distribution(nodes, graph):
    if not nodes:
       return 0
    degrees = [graph.degree(node) for node in nodes]
    return gini(np.array(degrees))

def calculate_atkinson_distribution(nodes, graph, epsilon=0.5):
    degrees = [graph.degree(node) for node in nodes]
    return #atkinson_index(np.array(degrees), epsilon)

def calculate_theil_distribution(nodes, graph):
    degrees = [graph.degree(node) for node in nodes]
    return #theil_index(np.array(degrees))

def combined_error(minority_nodes, majority_nodes, graph, method='gini', epsilon=0.5):
    if method == 'gini':
        error_minority = calculate_gini_distribution(minority_nodes, graph)
        error_majority = calculate_gini_distribution(majority_nodes, graph)
    elif method == 'atkinson':
        error_minority = calculate_atkinson_distribution(minority_nodes, graph, epsilon)
        error_majority = calculate_atkinson_distribution(majority_nodes, graph, epsilon)
    elif method == 'theil':
        error_minority = calculate_theil_distribution(minority_nodes, graph)
        error_majority = calculate_theil_distribution(majority_nodes, graph)
    else:
        raise ValueError("Unsupported method")

    return abs(error_minority - error_majority)

#%% Defione optimisation algo for seeding nodes

def assign_classes_iteratively(graph, method='gini', epsilon=0.5):
    nodes = list(graph.nodes())
    minority_nodes = []
    majority_nodes = []

    for node in nodes:
        # Temporarily assign the node to minority and calculate error
        temp_minority = minority_nodes + [node]
        temp_majority = majority_nodes

        error_minority = combined_error(temp_minority, temp_majority, graph, method, epsilon)
        
        # Temporarily assign the node to majority and calculate error
        temp_minority = minority_nodes
        temp_majority = majority_nodes + [node]

        error_majority = combined_error(temp_minority, temp_majority, graph, method, epsilon)

        # Assign the node to the group with the lower resulting error
        if error_minority < error_majority:
            minority_nodes.append(node)
        else:
            majority_nodes.append(node)

    return minority_nodes, majority_nodes

#%% Plotting

def plot_gini_distributions(minority_nodes, majority_nodes, graph):
    degrees_minority = [graph.degree(node) for node in minority_nodes]
    degrees_majority = [graph.degree(node) for node in majority_nodes]

    df_minority = pd.DataFrame(degrees_minority, columns=['degree'])
    df_majority = pd.DataFrame(degrees_majority, columns=['degree'])

    df_minority['degree_rank'] = df_minority['degree'].rank(method='min', ascending=False)
    df_majority['degree_rank'] = df_majority['degree'].rank(method='min', ascending=False)

    df_minority['class_label'] = 'minority'
    df_majority['class_label'] = 'majority'

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plot_gini_coefficient(df_minority, col_name='degree', class_label='class_label', title='Minority Gini Coefficient')
    
    plt.subplot(1, 2, 2)
    plot_gini_coefficient(df_majority, col_name='degree', class_label='class_label', title='Majority Gini Coefficient')
    
    plt.suptitle(f'Gini Coefficient: Minority = {gini(np.array(degrees_minority)):.2f}, Majority = {gini(np.array(degrees_majority)):.2f}')
    plt.show()



#%% Import graph

G = nx.erdos_renyi_graph(100, 0.5)


#%%

minority_nodes, majority_nodes = assign_classes_iteratively(G, method='gini')

for node in G.nodes():
    if node in minority_nodes:
        G.nodes[node]['class'] = 'minority'
    else:
        G.nodes[node]['class'] = 'majority'
        
plot_gini_distributions(minority_nodes, majority_nodes, G)