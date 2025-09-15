"""
Network Snapshots Visualization - Simplified
"""
import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import random

sys.path.append('../../')
sys.path.append('..')
import models_checks

def run_sims(n_runs, params, timesteps):
    """Run n simulations and return models + snapshots at specified timesteps"""
    models, snapshots = [], {t: [] for t in timesteps}
    
    # Get fixed initial states from first model
    np.random.seed(42)
    random.seed(42)
    ref_model = models_checks.simulate(0, params)
    init_states = {i: ref_model.graph.nodes[i]['agent'].state for i in ref_model.graph.nodes()}
    
    for i in range(n_runs):
        np.random.seed(42 + i * 1000)
        random.seed(42 + i * 1000)
        model = models_checks.simulate(i, params)
        
        # Apply fixed initial states and collect snapshots
        for t in timesteps:
            if t == 0:
                snapshot = init_states
            elif t < len(model.states):
                snapshot = {j: model.graph.nodes[j]['agent'].state for j in model.graph.nodes()}
            else:
                continue
            snapshots[t].append(snapshot)
        
        models.append(model)
    
    return models, snapshots

def create_avg_network(models, snapshots, t, threshold=0.3):
    """Create average network with edge frequency filtering"""
    n_nodes = len(models[0].graph.nodes())
    n_models = len(models)
    
    # Initial topology for marking rewired edges
    init_edges = set(models[0].graph.edges())
    if not nx.is_directed(models[0].graph):
        init_edges = {tuple(sorted(e)) for e in init_edges}
    
    # Average opinions
    avg_opinions = {}
    if t in snapshots and snapshots[t]:
        for i in range(n_nodes):
            opinions = [snap[i] for snap in snapshots[t] if i in snap]
            avg_opinions[i] = np.mean(opinions) if opinions else 0
    else:
        for i in range(n_nodes):
            avg_opinions[i] = np.mean([m.graph.nodes[i]['agent'].state for m in models])
    
    # Count edge frequencies
    edge_counts = {}
    for m in models:
        edges = set(m.graph.edges())
        if not nx.is_directed(m.graph):
            edges = {tuple(sorted(e)) for e in edges}
        for e in edges:
            edge_counts[e] = edge_counts.get(e, 0) + 1
    
    # Build average graph
    G = nx.DiGraph() if nx.is_directed(models[0].graph) else nx.Graph()
    G.add_nodes_from([(i, {'avg_opinion': avg_opinions[i]}) for i in range(n_nodes)])
    
    for e, count in edge_counts.items():
        freq = count / n_models
        if freq > threshold:
            is_rewired = e not in init_edges
            G.add_edge(e[0], e[1], weight=freq, rewired=is_rewired)
    
    return G

def plot_network(G, title="", ax=None, show_cbar=True, layout=None):
    """Plot network with largest component filtering"""
    # Filter to largest component
    if G.number_of_nodes() > 1:
        if nx.is_directed(G):
            comps = list(nx.weakly_connected_components(G))
        else:
            comps = list(nx.connected_components(G))
        
        if len(comps) > 1:
            largest = max(comps, key=len)
            G = G.subgraph(largest).copy()
    
    # Layout
    if layout is None:
        layout = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    else:
        layout = {n: pos for n, pos in layout.items() if n in G.nodes()}
    
    # Setup plot
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Node colors by opinion
    opinions = [G.nodes[n]['avg_opinion'] for n in G.nodes()]
    norm = Normalize(-1, 1)
    colors = plt.cm.coolwarm_r(norm(opinions))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, layout, node_color=colors, node_size=30, 
                          edgecolors='black', linewidths=0.5, ax=ax)
    
    # Draw edges with rewiring distinction
    if G.number_of_edges() > 0:
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        w_min, w_max = min(weights), max(weights)
        w_range = w_max - w_min if w_max > w_min else 1
        
        for u, v in G.edges():
            freq = G[u][v]['weight']
            is_rewired = G[u][v].get('rewired', False)
            width = 0.2 + (freq - w_min) / w_range * 1.5
            alpha = 0.3 + freq * 0.5
            color = plt.cm.plasma(freq) if is_rewired else plt.cm.gray(0.4 + freq * 0.4)
            
            nx.draw_networkx_edges(G, layout, [(u, v)], width=width, alpha=alpha,
                                 edge_color=[color], arrows=nx.is_directed(G), ax=ax)
    
    # Colorbar and styling
    if show_cbar and standalone:
        sm = ScalarMappable(cmap=plt.cm.coolwarm_r, norm=norm)
        plt.colorbar(sm, ax=ax, label='Opinion')
    
    ax.set_title(title)
    ax.axis('off')
    
    if standalone:
        plt.tight_layout()
        plt.show()
    
    return layout

def plot_evolution(models, snapshots, timesteps, params, threshold=0.3):
    """Plot network evolution across timesteps"""
    fig, axes = plt.subplots(1, len(timesteps), figsize=(6*len(timesteps), 6))
    if len(timesteps) == 1:
        axes = [axes]
    
    layout = None
    for i, (t, ax) in enumerate(zip(timesteps, axes)):
        G = create_avg_network(models, snapshots, t, threshold)
        show_cbar = (i == len(timesteps) - 1)
        layout = plot_network(G, f"t={t}", ax, show_cbar, layout)
    
    algo = params.get("rewiringAlgorithm", "")
    mode = params.get("rewiringMode", "")
    ntype = params.get("type", "")
    plt.suptitle(f"{ntype} - {algo} {mode} (n={len(models)})", fontsize=14)
    
    os.makedirs("../../Figs/Networks", exist_ok=True)
    plt.savefig(f"../../Figs/Networks/evolution_{ntype}_{algo}_{mode}.png", 
                bbox_inches='tight', dpi=300)
    plt.show()

def main():
    """Main function"""
    configs = {
        "cl": {"type": "cl", "nwsize": 100},
        "DPAH": {"type": "DPAH", "nwsize": 100}, 
        "FB": {"type": "FB", "nwsize": 786, "top_file": "FB_graph_N_786.gpickle"},
        "Twitter": {"type": "Twitter", "nwsize": 789, "top_file": "twitter_graph_N_789.gpickle"}
    }
    
    print("Networks:")
    for i, (k, v) in enumerate(configs.items()):
        print(f"{i}: {k} (N={v['nwsize']})")
    
    idx = int(input("Select: "))
    config = list(configs.values())[idx]
    n_runs = int(input("Runs (default 1): ") or "1")
    
    params = {
        "rewiringAlgorithm": "WTF", "rewiringMode": None, 
        "polarisingNode_f": 0.10, "timesteps": 15000, "plot": False,
        **config
    }
    
    total_t = params["timesteps"]
    timesteps = [0, total_t // 2, total_t - 1]
    threshold = 0.3 if config["type"] in ["cl", "FB"] else 0.2
    
    print(f"Running {n_runs} sims on {config['type']}...")
    models, snapshots = run_sims(n_runs, params, timesteps)
    
    # Debug info
    ref = models[0].graph
    print(f"Network: {ref.number_of_nodes()} nodes, {ref.number_of_edges()} edges")
    print(f"Avg degree: {2 * ref.number_of_edges() / ref.number_of_nodes():.1f}")
    
    plot_evolution(models, snapshots, timesteps, params, threshold)
    
    # Final network
    final_G = create_avg_network(models, snapshots, timesteps[-1], threshold)
    plot_network(final_G, f"Final - {config['type']} {params['rewiringAlgorithm']} {params['rewiringMode']}")
    
    return models, snapshots

if __name__ == "__main__":
    main()