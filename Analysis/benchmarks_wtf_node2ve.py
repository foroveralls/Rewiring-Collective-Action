# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:33:39 2024

@author: Jordan
"""

import models_checks_updated as models_checks
import time
from node2vec import Node2Vec
import numpy as np
import networkx as nx
import os
from fast_pagerank import pagerank_power


#%%
def timer(func, arg):
    start = time.time()
    func(arg)
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')

#%%
def wtf1(self, nodeIndex, topk=5):
    TOPK = topk
    nodes = self.graph.nodes()
    A = nx.to_scipy_sparse_matrix(self.graph, nodes, format='csr')
    num_cores = os.cpu_count()
    
    top = TOPK

    def _ppr(node_index, A, p, top):
        n = A.shape[0]
        pp = np.zeros(n)
        pp[node_index] = 1
        pr = pagerank_power(A, p=p, personalize=pp)
        pr_indices = np.argpartition(pr, -top-1)[-top-1:]
        pr_indices = pr_indices[np.argsort(pr[pr_indices])[::-1]]
        return pr_indices[pr_indices != node_index][:top]

    def get_circle_of_trust_per_node(A, p=0.85, top=top, num_cores=num_cores):
        return Parallel(n_jobs=num_cores, prefer="threads")(
            delayed(_ppr)(node_index, A, p, top) for node_index in range(A.shape[0])
        )

    def frequency_by_circle_of_trust(A, cot_per_node=None, p=0.85, top=10, num_cores=num_cores):
        if cot_per_node is None:
            cot_per_node = get_circle_of_trust_per_node(A, p, top, num_cores)
        unique_elements, counts_elements = np.unique(np.concatenate(cot_per_node), return_counts=True)
        count_dict = dict(zip(unique_elements, counts_elements))
        return [count_dict.get(node_index, 0) for node_index in range(A.shape[0])]

    def _salsa(node_index, cot, A, top=10):
        BG = nx.Graph()
        BG.add_nodes_from(['h{}'.format(vi) for vi in cot], bipartite=0)  # hubs
        edges = [('h{}'.format(vi), int(vj)) for vi in cot for vj in A[vi].indices]
        BG.add_nodes_from(set(e[1] for e in edges), bipartite=1)  # authorities
        BG.add_edges_from(edges)
        centrality = Counter({
            n: c for n, c in nx.eigenvector_centrality_numpy(BG).items()
            if isinstance(n, int) and n not in cot and n != node_index and n not in A[node_index].indices
        })
        return np.array([n for n, _ in centrality.most_common(top)])

    def frequency_by_who_to_follow(A, cot_per_node=None, p=0.85, top=top, num_cores=num_cores):
        if cot_per_node is None:
            cot_per_node = get_circle_of_trust_per_node(A, p, top, num_cores)
        results = Parallel(n_jobs=num_cores, prefer="threads")(
            delayed(_salsa)(node_index, cot, A, top) for node_index, cot in enumerate(cot_per_node)
        )
        unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
        count_dict = dict(zip(unique_elements, counts_elements))
        return [count_dict.get(node_index, 0) for node_index in range(A.shape[0])]

    def wtf_small(A, njobs):
        cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=TOPK, num_cores=njobs)
        wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=TOPK, num_cores=njobs)
        return wtf

    ranking = wtf_small(A, njobs=4)
    neighbour_index = np.argmax(ranking)
    
    
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

args = ({"type": "DPAH", "plot": False, "timesteps": 50, "rewiringAlgorithm": "node2vec"})
models_checks.nwsize = 20

changes = False #[node2vec_rewire, train_node2vec]

def test_runs(n):
    for i in range(n):
        print(i)
        
        model = models_checks.simulate(1, args, changes)
        #print(model.graph.number_of_nodes())

args_ = (1)
timer(test_runs, args_)

