# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:16:39 2024

@author: everall
"""

def wtf(self, nodeIndex):
    EXT = '.gpickle'
    TOPK = 10
    nodes = self.graph.nodes()
    A = nx.to_scipy_sparse_matrix(self.graph, nodes)
    
    def _ppr(node_index, A, p, top):
        pp = np.zeros(A.shape[0])
        pp[node_index] = A.shape[0]
        pr = pagerank_power(A, p=p, personalize=pp)
        pr = pr.argsort()[-top-1:][::-1]
        #time.sleep(0.01)
        return pr[pr!=node_index][:top]
    
    def get_circle_of_trust_per_node(A, p=0.85, top=10):
        return np.array([_ppr(node_index, A, p, top) for node_index in np.arange(A.shape[0])])
    
    def frequency_by_circle_of_trust(A, cot_per_node=None, p=0.85, top=10):
        results = cot_per_node if cot_per_node is not None else get_circle_of_trust_per_node(A, p, top)
        unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
        del(results)
        return [0 if node_index not in unique_elements else counts_elements[np.argwhere(unique_elements == node_index)[0, 0]] for node_index in np.arange(A.shape[0])]
    
    def _salsa(node_index, cot, A, top=10):
        BG = nx.Graph()
        BG.add_nodes_from(['h{}'.format(vi) for vi in cot], bipartite=0)  # hubs
        edges = [('h{}'.format(vi), int(vj)) for vi in cot for vj in np.argwhere(A[vi,:] != 0 )[:,1]]
        BG.add_nodes_from(set([e[1] for e in edges]), bipartite=1)  # authorities
        BG.add_edges_from(edges)
        centrality = Counter({n: c for n, c in nx.eigenvector_centrality_numpy(BG).items() if type(n) == int
                                                                                           and n not in cot
                                                                                           and n != node_index
                                                                                           and n not in np.argwhere(A[node_index,:] != 0 )[:,1] })
        del(BG)
        #time.sleep(0.01)
        return np.asarray([n for n, pev in centrality.most_common(top)])[:top]
    
    def frequency_by_who_to_follow(A, cot_per_node=None, p=0.85, top=10):
        cot_per_node = cot_per_node if cot_per_node is not None else get_circle_of_trust_per_node(A, p, top)
        results = np.array([_salsa(node_index, cot, A, top) for node_index, cot in enumerate(cot_per_node)])
        unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
        del(results)
        return [0 if node_index not in unique_elements else counts_elements[np.argwhere(unique_elements == node_index)[0, 0]] for node_index in np.arange(A.shape[0])]
    
    def wtf_small(A):
        cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=TOPK)
        wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=TOPK)
        return wtf
    
    return wtf_small(A)

