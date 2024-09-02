# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:53:20 2024

@author: Jordan
"""

def wtf1(self, nodeIndex, topk=5):
    TOPK = topk  # Reduce TOPK for better performance
    nodes = self.graph.nodes()
    A = nx.to_scipy_sparse_matrix(self.graph, nodes, format='csr')
    
    def _ppr(node_index, A, p, top):
        n = A.shape[0]
        pp = np.zeros(n)
        pp[node_index] = 1
        pr = pagerank_power(A, p=p, personalize=pp)
        pr_indices = np.argsort(pr)[-top-1:][::-1]
        return pr_indices[pr_indices != node_index][:top]

    def get_circle_of_trust_per_node(A, p=0.85, top=10, num_cores=40):
        return Parallel(n_jobs=num_cores, prefer="threads")(delayed(_ppr)(node_index, A, p, top) for node_index in range(A.shape[0]))
    
    def frequency_by_circle_of_trust(A, cot_per_node=None, p=0.85, top=10, num_cores=40):
        if cot_per_node is None:
            cot_per_node = get_circle_of_trust_per_node(A, p, top, num_cores)
        unique_elements, counts_elements = np.unique(np.concatenate(cot_per_node), return_counts=True)
        return [counts_elements[np.where(unique_elements == node_index)[0][0]] if node_index in unique_elements else 0 for node_index in range(A.shape[0])]
    
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
    
    def frequency_by_who_to_follow(A, cot_per_node=None, p=0.85, top=10, num_cores=40):
        if cot_per_node is None:
            cot_per_node = get_circle_of_trust_per_node(A, p, top, num_cores)
        results = Parallel(n_jobs=num_cores, prefer="threads")(delayed(_salsa)(node_index, cot, A, top) for node_index, cot in enumerate(cot_per_node))
        unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
        return [counts_elements[np.where(unique_elements == node_index)[0][0]] if node_index in unique_elements else 0 for node_index in range(A.shape[0])]
    
    def who_to_follow_rank(A, njobs=1):
        return wtf_small(A, njobs)
    
    def wtf_small(A, njobs):
        cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=TOPK, num_cores=njobs)
        wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=TOPK, num_cores=njobs)
        return wtf
    
    ranking = wtf_small(A, njobs=4)
    neighbour_index = ranking.index(max(ranking))
    self.rewire(nodeIndex, neighbour_index)