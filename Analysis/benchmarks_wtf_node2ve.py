# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:33:39 2024

@author: Jordan
"""

#%%
import os 
import sys
# Ensure the parent directory is in the sys.path for auxiliary imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
#%%
import models_checks_updated as models_checks
import time
from node2vec import Node2Vec
import numpy as np
import networkx as nx
#from fast_wtf import frequency_by_who_to_follow as frequency_wtf
from fast_wtf import wtf_full
from joblib import Parallel, delayed
from fast_pagerank import pagerank_power
from scipy.sparse import csr_matrix
#%%
def timer(func, arg):
    start = time.time()
    func(arg)
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')

#%%


def wtf1(self, nodeIndex):
    print("new")
    num_cores = os.cpu_count()
    A= nx.to_scipy_sparse_array(self.graph, format='csr')

    # Convert scipy.sparse matrix to numpy arrays for data, indices, and indptr
    data = np.array(A.data, dtype=np.float64)
    indices = np.array(A.indices, dtype=np.int32)
    indptr = np.array(A.indptr, dtype=np.int32)
    
    ranking = wtf_full(data, indices, indptr, A.shape[0], A.shape[1], topk=10, num_cores=num_cores-1)
    
    neighbour_index = np.argmax(ranking)
        
    
    neighbours = list(self.graph.adj[nodeIndex].keys())
    self.rewire(nodeIndex, neighbour_index)
    self.break_link(nodeIndex, neighbour_index, neighbours)
    
    
    
    
# def wtf1(self, nodeIndex, topk=5):
#     print("new")
#     TOPK = topk
#     nodes = self.graph.nodes()
#     A = nx.to_scipy_sparse_matrix(self.graph, nodes, format='csr')
#     #A = nx.to_numpy_array(self.graph)
#     num_cores = os.cpu_count()
#     alpha = 0.85

#     def _ppr(node_index, A, p, top):
#         n = A.shape[0]
#         pp = np.zeros(n)
#         pp[node_index] = 1
#         pr = pagerank_power(A, p=p, personalize=pp)
#         pr_indices = np.argpartition(pr, -top-1)[-top-1:]
#         pr_indices = pr_indices[np.argsort(pr[pr_indices])[::-1]]
#         return pr_indices[pr_indices != node_index][:top]

#     def get_circle_of_trust_per_node(A, p=0.85, top=TOPK, num_cores=num_cores):
#         return Parallel(n_jobs=num_cores, prefer="threads")(
#             delayed(_ppr)(node_index, A, p, top) for node_index in range(A.shape[0])
#         )

#     def wtf_full(A, njobs):
#         cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=TOPK, num_cores=njobs)
#         A = csr_matrix.toarray(A)
#         wtf = frequency_wtf(A, cot_per_node,  alpha, TOPK, num_cores)
#         return wtf
    

#     ranking = wtf_full(A, njobs = num_cores)
#     neighbour_index = np.argmax(ranking)
#     neighbours = list(self.graph.adj[nodeIndex].keys())
#     self.rewire(nodeIndex, neighbour_index)
#     self.break_link(nodeIndex, neighbour_index, neighbours)
    
        

    
    
def train_node2vec(self):
    
    # Generate walks
    node2vec = Node2Vec(self.graph, dimensions=64, walk_length=30, num_walks=150, workers=4, quiet = True)
    
    # Train model
    self.model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    
   
        
def node2vec_rewire(self, nodeIndex):
     
    most_similar_i = self.model.wv.most_similar(str(nodeIndex))[0][0]
    
    #sim = int(get_similar_agents(nodeIndex)[0][0])
    
    print("sim_ind: ", most_similar_i, nodeIndex)
    self.rewire(nodeIndex, int(most_similar_i))




#%%

args = ({"type": "DPAH", "plot": False, "timesteps": 2, "rewiringAlgorithm": "wtf"})
models_checks.nwsize = 500

changes =  False #[wtf1] #False # [node2vec_rewire, train_node2vec] #False

def test_runs(n):
    for i in range(n):
        print(i)
        
        model = models_checks.simulate(1, args, changes)
        #print(model.graph.number_of_nodes())

args_ = (1)
timer(test_runs, args_)

