# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:51:54 2024

@author: Jordan
"""

import subprocess
import numpy as np
import os


#%%

def save_graph_as_edgelist(graph, filename):
    with open(filename, 'w') as f:
        for edge in graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")


def get_node2vec_path():
    # Get the directory of the current script (functions.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the node2vec executable, cover case for linux + windows
    node2vec_file = "node2vec.exe" if os.name in "nt" else "node2vec" 
       
    node2vec_executable = os.path.join(script_dir, node2vec_file)
    return node2vec_executable

def validate_embeddings_file(filepath, graph_nodes, expected_dimensions=64):
    valid_nodes = set(graph_nodes)
    cleaned_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            values = line.strip().split()
            node = int(values[0])
            if node in valid_nodes and len(values) == expected_dimensions + 1:
                cleaned_lines.append(line.strip())
            else:
                print(f"Invalid line detected and removed: {line.strip()}")
    
    with open(filepath, 'w') as f:
        for line in cleaned_lines:
            f.write(f"{line}\n")

def run_node2vec(node2vec_executable, input_file, output_file, dimensions=64, walk_length=50, num_walks=5, context_size=10):
    if not os.path.exists(node2vec_executable):
        raise FileNotFoundError(f"node2vec executable not found at {node2vec_executable}. Please compile it from the SNAP repository.")

    command = [
        node2vec_executable,
        f'-i:{input_file}',
        f'-o:{output_file}',
        f'-d:{dimensions}',
        f'-l:{walk_length}',
        f'-r:{num_walks}',
        f'-k:{context_size}'
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_embeddings(filepath, expected_dimensions=64):
    embeddings = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Always remove the last line
        lines = lines[1:]  # Remove the last line

        for line in lines:
            values = line.strip().split()
            node = int(values[0])  # Ensure node indices are integers
            vector = np.array([float(x) for x in values[1:]])
            if vector.shape[0] != expected_dimensions:
                print(f"Debug: Line causing issue: {line.strip()}")
                raise ValueError(f"Embedding for node {node} has incorrect dimensions: {vector.shape[0]}")
            embeddings[node] = vector
    return embeddings